"""Read-only native WebShop telemetry plus the shared safe checkpoint wrapper."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from common import atomic_json, protocol
from checkpoint_io import NativeCheckpointMixin
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def native_validation_selection(env, base_seed):
    """Preview native RandomState.choice without advancing its actual state."""
    preview = np.random.RandomState()
    preview.set_state(env._rng.get_state())
    indices = np.repeat(preview.choice(env.goal_idxs, size=env.env_num, replace=False), env.group_n).tolist()
    return [{'worker': i, 'worker_seed': base_seed + i // env.group_n, 'goal_index': int(index)}
            for i, index in enumerate(indices)]


class WebShopTrainer(NativeCheckpointMixin, RayPPOTrainer):
    def fit(self):
        from verl.utils.tracking import Tracking
        original = Tracking.log
        destination = Path(os.environ['GRPO_RUN_DIR']) / 'metrics.jsonl'

        def capture(tracker, data, step, backend=None):
            original(tracker, data=data, step=step, backend=backend)
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'step': int(step), 'metrics': data},
                    default=lambda x: x.item(), allow_nan=False) + '\n')
                stream.flush()

        Tracking.log = capture
        try:
            return super().fit()
        finally:
            Tracking.log = original

    def _validate(self):
        original_loop = self.traj_collector.multi_turn_loop
        native_env = self.val_envs.envs
        original_reset = native_env.reset
        records, selections = [], []

        def reset():
            selections.append(native_validation_selection(native_env, self.config.env.seed + 1000))
            return original_reset()

        def capture(*args, **kwargs):
            batch = original_loop(*args, **kwargs)
            ids = batch.batch['responses']
            mask = batch.batch['attention_mask'][:, -ids.shape[-1]:].bool()
            nt = batch.non_tensor_batch
            for i in range(len(ids)):
                tokens = ids[i][mask[i]].cpu().tolist()
                text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                records.append({'trajectory_id': str(nt['traj_uid'][i]), 'step': int(nt['turn_step'][i]),
                    'episode_reward': float(nt['episode_rewards'][i]), 'text': text,
                    'token_ids': tokens, 'action_valid': bool(nt['is_action_valid'][i]),
                    'protocol': protocol(text)})
            return batch

        native_env.reset = reset
        self.traj_collector.multi_turn_loop = capture
        try:
            metrics = super()._validate()
        finally:
            native_env.reset = original_reset
            self.traj_collector.multi_turn_loop = original_loop
        episodes = {}
        for row in records:
            previous = episodes.setdefault(row['trajectory_id'], row['episode_reward'])
            if previous != row['episode_reward']:
                raise RuntimeError('Inconsistent native trajectory reward')
        expected = int(self.config.data.val_batch_size)
        if len(episodes) != expected or len(selections) != 1 or len(selections[0]) != expected:
            raise RuntimeError('Expected one native WebShop validation batch')
        successes = sum(reward > 0 for reward in episodes.values())
        if abs(metrics['val/success_rate'] - successes / expected) > 1e-6:
            raise RuntimeError('Episode success differs from native WebShop metric')
        schedule = selections[0]
        schedule_sha = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
        score_key = 'val/webshop_task_score (not success_rate)'
        summary = {'update': int(self.global_steps), 'episodes': expected,
            'successes': successes, 'success_rate': successes / expected,
            'task_score': float(metrics[score_key]), 'task_score_0_to_100': 100 * float(metrics[score_key]),
            'selection_sha256': schedule_sha, 'native_worker_goal_indices': schedule,
            'decisions': len(records), 'native_metrics': metrics,
            'reasoning_open_rate': sum(r['protocol']['reasoning_open'] for r in records) / len(records),
            'strict_protocol_rate': sum(r['protocol']['strict_protocol'] for r in records) / len(records),
            'native_action_valid_rate': sum(r['action_valid'] for r in records) / len(records),
            'task_id_note': 'Native per-worker shuffled goal indices, not globally canonical human test IDs'}
        out = Path(os.environ['GRPO_RUN_DIR']) / 'evaluation' / f'update_{self.global_steps:03d}'
        atomic_json(out / 'summary.json', summary)
        atomic_json(out / 'decisions.json', records)
        return metrics
