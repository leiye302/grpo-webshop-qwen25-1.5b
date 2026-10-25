"""Read-only native WebShop telemetry plus the shared safe checkpoint wrapper."""
import json
import os
from pathlib import Path
from common import atomic_json, protocol
from checkpoint_io import NativeCheckpointMixin
from fixed_validation import (manifest, native_validation_selection,
                              reset_fixed_validation, record_identity)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


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
        records, selections, identities = [], [], []

        def reset():
            fixed = manifest()
            if identities:
                raise RuntimeError('Expected exactly one140-task validation reset')
            result, identity = reset_fixed_validation(native_env, original_reset, self.config.env.seed + 1000)
            record_identity(os.environ['GRPO_RUN_DIR'], identity)
            selections.extend(fixed['selection'])
            identities.append(identity)
            return result

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
        expected = manifest()['count']
        if len(episodes) != expected or len(selections) != expected or len(identities) != 1:
            raise RuntimeError('Expected all140 WebShop validation trajectories')
        successes = sum(reward > 0 for reward in episodes.values())
        if abs(metrics['val/success_rate'] - successes / expected) > 1e-6:
            raise RuntimeError('Episode success differs from native WebShop metric')
        schedule = selections
        identity = identities[0]
        score_key = 'val/webshop_task_score (not success_rate)'
        summary = {'update': int(self.global_steps), 'episodes': expected,
            'successes': successes, 'success_rate': successes / expected,
            'task_score': float(metrics[score_key]), 'task_score_0_to_100': 100 * float(metrics[score_key]),
            'selection_sha256': identity['selection_sha256'],
            'fixed_validation_identity': identity, 'native_worker_goal_indices': schedule,
            'decisions': len(records), 'native_metrics': metrics,
            'reasoning_open_rate': sum(r['protocol']['reasoning_open'] for r in records) / len(records),
            'strict_protocol_rate': sum(r['protocol']['strict_protocol'] for r in records) / len(records),
            'native_action_valid_rate': sum(r['action_valid'] for r in records) / len(records),
            'task_id_note': 'Fixed140 synthetic objectives; content IDs excluded from every training worker; not canonical human500'}
        out = Path(os.environ['GRPO_RUN_DIR']) / 'evaluation' / f'update_{self.global_steps:03d}'
        atomic_json(out / 'summary.json', summary)
        atomic_json(out / 'decisions.json', records)
        return metrics
