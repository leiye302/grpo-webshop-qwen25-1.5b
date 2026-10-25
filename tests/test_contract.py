import ast
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace, ModuleType
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import UPSTREAM, WORLD_SIZE, devices, load, protocol
from run import assert_grpo_only, config
from setup import verify


def test_native_grpo_and_webshop_hash_locks():
    lock = verify()
    assert len(lock['unchanged_files']) >= 30
    assert lock['native_launcher'] == 'examples/grpo_trainer/run_webshop_8gpu.sh'
    assert set(lock['patched_files']) == {'verl/trainer/main_ppo.py',
        'agent_system/environments/env_package/webshop/envs.py'}
    for name in ['agent_system/environments/env_manager.py',
                 'agent_system/environments/env_package/webshop/projection.py',
                 'agent_system/environments/prompts/webshop.py']:
        assert name in lock['unchanged_files']


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_native_webshop_config(size, tmp_path):
    cfg = config(size, tmp_path/'model', tmp_path/'run')
    assert cfg.trainer.n_gpus_per_node == cfg.ray_init.num_gpus == WORLD_SIZE == 4
    assert cfg.ray_init.num_cpus == 32
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.data.val_batch_size == 140 and cfg.env.env_name == 'Webshop'
    assert cfg.env.webshop.use_small and not cfg.env.webshop.human_goals
    assert cfg.env.max_steps == 15 and cfg.env.history_length == 2
    assert cfg.data.max_prompt_length == 4096 and cfg.data.max_response_length == 512
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 8
    assert cfg.actor_rollout_ref.actor.use_kl_loss
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == .01
    assert cfg.actor_rollout_ref.actor.entropy_coeff == .001
    assert cfg.trainer.total_training_steps == cfg.trainer.total_epochs == 150
    assert cfg.trainer.test_freq == 5 and cfg.trainer.save_freq == 20
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.actor_rollout_ref.rollout.n == 1
    assert cfg.actor_rollout_ref.model.get('lora_rank', 0) == 0


@pytest.mark.parametrize('name', ['sdar', 'skillsd', 'rlsd', 'opsd', 'opd', 'local_hints'])
def test_distillation_rejected(name, tmp_path):
    from omegaconf import OmegaConf
    cfg = config('1.5b', tmp_path/'model', tmp_path/'run')
    OmegaConf.update(cfg, 'algorithm.' + name, {}, force_add=True)
    with pytest.raises(RuntimeError, match='Unexpected distillation'):
        assert_grpo_only(cfg)


@pytest.mark.parametrize('name', ['use_sdar_loss', 'use_sdl_loss', 'use_opd_loss'])
def test_actor_auxiliary_loss_rejected(name, tmp_path):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    OmegaConf.update(cfg, 'actor_rollout_ref.actor.' + name, True, force_add=True)
    with pytest.raises(RuntimeError, match='auxiliary distillation'):
        assert_grpo_only(cfg)


@pytest.mark.parametrize('spec', ['0,1', '0,1,2', '0,1,2,2', '0,1,2,3,4', '', '0,1,2,GPU-id'])
def test_invalid_device_allocation(spec):
    with pytest.raises(ValueError):
        devices(spec)


def test_device_binding_and_protocol_telemetry():
    assert devices('4,5,6,7') == ['4','5','6','7']
    assert protocol('<think>Search first.</think><action>search[shoes]</action>')['strict_protocol']
    assert not protocol('<think></think><action>click[buy now]</action>')['strict_protocol']
    assert not protocol('<think>Plan</think><action>click[a]</action><action>click[b]</action>')['strict_protocol']


def test_native_training_entry_does_not_instantiate_teacher():
    source = (ROOT/'src/run.py').read_text()
    assert 'from verl.trainer.main_ppo import run_ppo' in source and 'run_ppo(cfg)' in source
    main = (UPSTREAM/'verl/trainer/main_ppo.py').read_text()
    assert 'from grpo_eval import WebShopTrainer as RayPPOTrainer' in main
    assert 'TeacherPolicy' not in main and 'SkillProvider' not in main
    tree = ast.parse((UPSTREAM/'verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    fit = ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit'))
    assert 'teacher_log_probs' not in fit and 'teacher_forward' not in fit


def test_native_grpo_loss_has_real_gradient_and_optimizer_update():
    torch = pytest.importorskip('torch')
    import numpy as np
    source = ROOT/'.deps/sdar/verl/trainer/ppo/core_algos.py'
    tree = ast.parse(source.read_text())
    names = {'compute_grpo_outcome_advantage', 'agg_loss', 'compute_policy_loss'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == 3
    namespace = {'torch': torch, 'np': np, 'defaultdict': defaultdict,
                 'verl_F': SimpleNamespace(masked_mean=lambda v,m:(v*m).sum()/m.sum())}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    mask = torch.tensor([[1.,1.,0.]]*4)
    rewards = torch.tensor([[0.,10.,0.], [0.,0.,0.], [0.,10.,0.], [0.,0.,0.]])
    advantages, _ = namespace['compute_grpo_outcome_advantage'](
        rewards, mask, np.array(['a','a','b','b']), np.array(['a0','a1','b0','b1']))
    assert torch.all(advantages[[0,2],:2] > 0)
    assert torch.all(advantages[[1,3],:2] < 0)
    assert not advantages.requires_grad and torch.count_nonzero(advantages[:,2]) == 0
    student = torch.nn.Parameter(torch.full_like(mask, -1.0))
    old = student.detach().clone()
    optimizer = torch.optim.SGD([student], lr=0.1)
    loss = namespace['compute_policy_loss'](old, student, advantages, mask, cliprange=0.2)[0]
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[0,0] < 0 and student.grad[1,0] > 0
    assert torch.count_nonzero(student.grad[:,2]) == 0
    optimizer.step()
    assert student[0,0] > old[0,0] and student[1,0] < old[1,0]
    assert torch.equal(student[:,2], old[:,2])



def test_scientific_settings_match_official_except_resources_and_validation(tmp_path):
    from omegaconf import OmegaConf
    source = (ROOT/'.deps/sdar/examples/grpo_trainer/run_webshop_8gpu.sh').read_text()
    assignments = dict(re.findall(r'(?m)^\s*\+?([\w.]+)=([^\s\\]+)', source))
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    keys = [
        'algorithm.adv_estimator', 'data.train_batch_size', 'data.max_prompt_length',
        'data.max_response_length', 'data.filter_overlong_prompts', 'data.truncation',
        'data.return_raw_chat', 'actor_rollout_ref.actor.optim.lr',
        'actor_rollout_ref.model.use_remove_padding',
        'actor_rollout_ref.actor.ppo_mini_batch_size',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
        'actor_rollout_ref.actor.use_kl_loss', 'actor_rollout_ref.actor.kl_loss_coef',
        'actor_rollout_ref.actor.kl_loss_type',
        'actor_rollout_ref.model.enable_gradient_checkpointing',
        'actor_rollout_ref.actor.fsdp_config.param_offload',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',
        'actor_rollout_ref.rollout.tensor_model_parallel_size',
        'actor_rollout_ref.rollout.gpu_memory_utilization',
        'actor_rollout_ref.rollout.enable_chunked_prefill',
        'actor_rollout_ref.rollout.enforce_eager',
        'actor_rollout_ref.rollout.free_cache_engine',
        'actor_rollout_ref.rollout.val_kwargs.temperature',
        'actor_rollout_ref.rollout.val_kwargs.do_sample',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu',
        'actor_rollout_ref.ref.fsdp_config.param_offload',
        'actor_rollout_ref.actor.use_invalid_action_penalty',
        'actor_rollout_ref.actor.invalid_action_penalty_coef',
        'algorithm.use_kl_in_reward', 'env.env_name', 'env.seed', 'env.max_steps',
        'env.rollout.n', 'trainer.critic_warmup',
        'trainer.ray_wait_register_center_timeout', 'trainer.nnodes',
        'trainer.total_epochs', 'trainer.val_before_train', 'trainer.test_freq', 'env.resources_per_worker.num_cpus',
    ]
    for key in keys:
        if key not in assignments:
            continue  # Values inherited unchanged from the pinned native YAML.
        raw = assignments[key]
        if raw.startswith('$'):
            raw = assignments[raw[1:]]
        if raw.lower() in ('true', 'false'):
            expected = raw.lower() == 'true'
        elif raw.startswith("'"):
            expected = raw.strip("'")
        else:
            try:
                expected = float(raw)
            except ValueError:
                expected = raw
        assert OmegaConf.select(cfg, key) == expected, key



def test_global_minibatch_preserved_by_native_four_rank_accumulation(tmp_path):
    cfg = config('3b', tmp_path/'model', tmp_path/'run')
    source = UPSTREAM/'verl/workers/fsdp_workers.py'
    tree = ast.parse(source.read_text())
    normalization = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'self._is_actor'
        and 'self.config.actor.ppo_mini_batch_size *= self.config.rollout.n' in ast.unparse(n))
    output = {}
    for ranks in (4, 8):
        worker = SimpleNamespace(_is_actor=True, config=copy.deepcopy(cfg.actor_rollout_ref),
            device_mesh=SimpleNamespace(size=lambda:ranks), ulysses_sequence_parallel_size=1)
        exec(compile(ast.Module(body=[normalization], type_ignores=[]), str(source), 'exec'), {'self':worker})
        local = worker.config.actor.ppo_mini_batch_size
        micro = worker.config.actor.ppo_micro_batch_size_per_gpu
        assert local * ranks == 64
        output[ranks] = (local, micro, local//micro)
    assert output == {4:(16,8,2), 8:(8,8,1)}


def test_sizes_have_identical_scientific_settings(tmp_path):
    from omegaconf import OmegaConf
    configs = [config(size, tmp_path/('model'+size), tmp_path/('run'+size)) for size in ('1.5b','3b')]
    for cfg in configs:
        cfg.actor_rollout_ref.model.path = 'model'
        cfg.critic.model.tokenizer_path = 'model'
        cfg.reward_model.model.input_tokenizer = 'model'
        cfg.trainer.experiment_name = 'experiment'
        cfg.data.train_files = cfg.data.val_files = 'parquet'
        cfg.trainer.default_local_dir = cfg.trainer.validation_data_dir = cfg.trainer.rollout_data_dir = 'output'
        cfg.ray_init.runtime_env.env_vars.GRPO_RUN_DIR = 'output'
    assert OmegaConf.to_container(configs[0], resolve=True) == OmegaConf.to_container(configs[1], resolve=True)


def evaluation_module(monkeypatch, Native):
    native = ModuleType('verl.trainer.ppo.ray_trainer')
    native.RayPPOTrainer = Native
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.ray_trainer', native)
    spec = importlib.util.spec_from_file_location('test_webshop_eval', ROOT/'src/grpo_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_preview_preserves_actual_rng(monkeypatch):
    module = evaluation_module(monkeypatch, object)
    env = SimpleNamespace(_rng=np.random.RandomState(1000), goal_idxs=range(500), env_num=128, group_n=1)
    before = copy.deepcopy(env._rng.get_state())
    preview = module.native_validation_selection(env, 1000)
    actual = env._rng.choice(env.goal_idxs, size=128, replace=False).tolist()
    assert [r['goal_index'] for r in preview] == actual
    assert [r['worker_seed'] for r in preview] == list(range(1000,1128))
    fresh = np.random.RandomState()
    fresh.set_state(before)
    assert fresh.choice(500,128,replace=False).tolist() == actual


@pytest.mark.parametrize('failure', [False, True])
def test_readonly_validation_hook_matches_native_metrics_and_restores(tmp_path, monkeypatch, failure):
    torch = pytest.importorskip('torch')
    from fixed_validation import manifest
    keys = [r['task_id'] for r in manifest()['selection']]
    batch = SimpleNamespace(batch={'responses':torch.ones((140,2),dtype=torch.long),
        'attention_mask':torch.ones((140,4),dtype=torch.long)},
        non_tensor_batch={'traj_uid':[str(i) for i in range(140)], 'turn_step':[0]*140,
            'episode_rewards':[10.]*70+[0.]*70, 'is_action_valid':[True]*140})
    metrics = {'val/success_rate':.5, 'val/webshop_task_score (not success_rate)':.7}
    class Native:
        def _validate(self):
            self.val_envs.envs.reset()
            self.traj_collector.multi_turn_loop()
            if failure:
                raise RuntimeError('native failure')
            return metrics
    module = evaluation_module(monkeypatch, Native)
    env = SimpleNamespace(_rng=np.random.RandomState(1000), goal_idxs=range(140),
                          env_num=140, group_n=1, is_train=False)
    def original_reset():
        indices = env._rng.choice(env.goal_idxs, 140, replace=False)
        return [f'goal-{i}' for i in indices], [{'task_id': keys[i]} for i in indices]
    original_loop = lambda: batch
    env.reset = original_reset
    trainer = module.WebShopTrainer()
    trainer.global_steps = 5
    trainer.config = config('1.5b', tmp_path/'model', tmp_path/'run')
    trainer.val_envs = SimpleNamespace(envs=env)
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=original_loop)
    trainer.tokenizer = SimpleNamespace(decode=lambda ids,**kw:'<think>Plan.</think><action>search[shoes]</action>')
    monkeypatch.setenv('GRPO_RUN_DIR', str(tmp_path/'run'))
    if failure:
        with pytest.raises(RuntimeError, match='native failure'):
            trainer._validate()
    else:
        assert trainer._validate() == metrics
        out = json.loads((tmp_path/'run/evaluation/update_005/summary.json').read_text())
        assert out['successes'] == 70 and out['episodes'] == 140
        assert out['task_score'] == .7 and out['task_score_0_to_100'] == 70
        assert out['reasoning_open_rate'] == out['strict_protocol_rate'] == 1
        assert len(out['native_worker_goal_indices']) == 140
    assert env.reset is original_reset
    assert trainer.traj_collector.multi_turn_loop is original_loop
    if not failure:
        trainer.global_steps = 10
        trainer._validate()
        later = json.loads((tmp_path/'run/evaluation/update_010/summary.json').read_text())
        assert later['fixed_validation_identity'] == out['fixed_validation_identity']


def test_fixed_validation_covers_all140_across_checkpoints_and_resume():
    from fixed_validation import reset_fixed_validation, manifest
    selected = []
    keys = [r['task_id'] for r in manifest()['selection']]
    env = SimpleNamespace(_rng=np.random.RandomState(1000), goal_idxs=range(140),
                          env_num=140, group_n=1, is_train=False)
    def reset():
        indices = env._rng.choice(env.goal_idxs, 140, replace=False).tolist()
        selected.append(indices)
        return [f'task-{i}' for i in indices], [{'task_id': keys[i]} for i in indices]
    identities = []
    for _ in range(3):
        env._rng.choice(140, 140, replace=False)  # A progressed/private native RNG is irrelevant.
        before = copy.deepcopy(env._rng.get_state())
        _, identity = reset_fixed_validation(env, reset, 1000)
        identities.append(identity)
        after = env._rng.get_state()
        assert before[0] == after[0] and np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    expected = [r['goal_index'] for r in manifest()['selection']]
    assert selected == [expected]*3 and expected == list(range(140))
    assert identities == [identities[0]]*3 and len(set(identities[0]['task_ids'])) == 140
    env._rng = np.random.RandomState(1000)  # Recreated environment on resume.
    _, restarted = reset_fixed_validation(env, reset, 1000)
    assert restarted == identities[0]
    env.is_train = True
    with pytest.raises(RuntimeError, match='validation environment'):
        reset_fixed_validation(env, reset, 1000)


def test_actual_native_vector_reset_frozen_and_training_random_on_filtered_pool(monkeypatch):
    from fixed_validation import reset_fixed_validation, manifest
    source = UPSTREAM/'agent_system/environments/env_package/webshop/envs.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebshopMultiProcessEnv')
    class Worker:
        def __init__(self, seed, kwargs, is_train): self.seed, self.is_train = seed, is_train
        def get_goals(self): return list(range(1000 if self.is_train else 140))
        def reset(self, index):
            key = 'training' if self.is_train else manifest()['selection'][index]['task_id']
            return f'{self.seed}:{index}', {'task_id': key}
        def close(self): pass
    class Handle:
        def __init__(self, obj): self.obj = obj
        def __getattr__(self, name): return SimpleNamespace(remote=getattr(self.obj, name))
    remote = SimpleNamespace(remote=lambda *a:Handle(Worker(*a)))
    ray = SimpleNamespace(is_initialized=lambda:True, remote=lambda **kw:lambda c:remote,
                          get=lambda x:x, kill=lambda x:None)
    namespace = {'gym':SimpleNamespace(Env=object), 'np':np, 'ray':ray, 'WebshopWorker':Worker}
    monkeypatch.setitem(sys.modules, 'webshop_worker', SimpleNamespace(DisjointWebshopWorker=Worker))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
    Native = namespace['WebshopMultiProcessEnv']
    training = Native(0,16,8,{},is_train=True)
    validation = Native(1000,140,1,{},is_train=False)
    try:
        first_train = training.reset()
        val1, identity1 = reset_fixed_validation(validation, validation.reset, 1000)
        val2, identity2 = reset_fixed_validation(validation, validation.reset, 1000)
        assert val1 == val2 and identity1 == identity2
        assert val1[0] == [f"{r['worker_seed']}:{r['goal_index']}" for r in manifest()['selection']]
        second_train = training.reset()
        control = np.random.RandomState(0)
        first = np.repeat(control.choice(range(1000),16,replace=False),8)
        second = np.repeat(control.choice(range(1000),16,replace=False),8)
        assert first_train[0] == [f'{i//8}:{x}' for i,x in enumerate(first)]
        assert second_train[0] == [f'{i//8}:{x}' for i,x in enumerate(second)]
        assert first_train != second_train
    finally:
        training.close()
        validation.close()


def test_changed_initial_goal_is_rejected_not_silently_compared(tmp_path):
    from fixed_validation import record_identity
    identity = {'manifest_sha256':'same', 'initial_observations_sha256':'original'}
    record_identity(tmp_path, identity)
    record_identity(tmp_path, identity)
    with pytest.raises(RuntimeError, match='initial tasks changed'):
        record_identity(tmp_path, dict(identity, initial_observations_sha256='different'))


def test_published140_manifest_diversity_and_content_ids():
    from fixed_validation import manifest
    from task_split import manifest as split_manifest
    split, fixed = split_manifest(), manifest()
    assert fixed['count'] == fixed['batch_size'] == 140
    assert fixed['task_split_sha256'] == load('upstream.json')['task_split_sha256']
    assert len(set(split['validation_task_ids'])) == 140
    assert split['diversity']['category_counts'] == dict.fromkeys(
        ['beauty', 'electronics', 'fashion', 'garden', 'grocery'], 28)
    assert split['diversity']['distinct_products'] == 140
    assert split['diversity']['distinct_queries'] == 123
    assert split['diversity']['distinct_product_categories'] == 126
    assert split['diversity']['option_counts'] == {'0':55, '1':49, '2':29, '3':7}
    assert split['training_goal_entries'] + split['validation_goal_entries_before_dedup'] == split['raw_goal_entries']
    assert split['raw_goal_entries'] == 6910 and split['training_goal_entries'] == 6765


@pytest.fixture
def synthetic_split(monkeypatch):
    import task_split as split
    goals = []
    for category in split.CATEGORIES:
        for index in range(40):
            goals.append({'asin': f'{category}-{index}', 'name': f'{category} item{index}',
                'category': category, 'query': f'query-{index%12}',
                'product_category': f'{category}/subcategory-{index%15}',
                'instruction_text': f'Find {category} item{index}, and price lower than 50.00 dollars',
                'price_upper': 50., 'weight': 1.,
                'attributes': [f'attribute-{i}' for i in range(2+index%4)],
                'goal_options': {f'option-{i}':'value' for i in range(index%4)}})
    value = split.make_manifest(goals)
    monkeypatch.setattr(split, 'manifest', lambda: value)
    split.validation_ids.cache_clear()
    try:
        yield split, goals, value
    finally:
        split.validation_ids.cache_clear()


def test_balanced_selection_independent_of_native_order_and_price_rng(synthetic_split):
    split, goals, value = synthetic_split
    reversed_goals = copy.deepcopy(goals[::-1])
    for goal in reversed_goals:
        goal['price_upper'] = 90.
        goal['instruction_text'] = goal['instruction_text'].replace('50.00', '90.00')
    repeated = split.make_manifest(reversed_goals)
    assert value['validation_task_ids'] == repeated['validation_task_ids']
    assert value['diversity'] == repeated['diversity']
    before = np.random.RandomState(2).get_state()
    random_state = __import__('random').getstate()
    split.make_manifest(goals)
    assert __import__('random').getstate() == random_state


def test_all_worker_orders_exclude_same140_and_preserve_other_entries(synthetic_split):
    split, goals, value = synthetic_split
    val = set(value['validation_task_ids'])
    for seed in (0,1,15,1000,1139):
        order = np.random.RandomState(seed).permutation(len(goals))
        rows = [copy.deepcopy(goals[i]) for i in order]
        for goal in rows:
            goal['instruction_text'] = goal['instruction_text'].replace('50.00', f'{60+seed%30}.00')
        training, validation = split.partition_goals(rows, True), split.partition_goals(rows, False)
        train_ids = {split.task_identity(g) for g in training}
        assert not train_ids & val
        assert train_ids | val == {split.task_identity(g) for g in goals}
        assert [split.task_identity(g) for g in validation] == value['validation_task_ids']
        assert training == [g for g in rows if split.task_identity(g) not in val]


def test_price_and_duplicate_product_alias_cannot_leak_into_training(synthetic_split):
    split, goals, value = synthetic_split
    first = next(g for g in goals if split.task_identity(g) == value['validation_task_ids'][0])
    alias = dict(copy.deepcopy(first), asin='alias-product', price_upper=70.)
    alias['instruction_text'] = alias['instruction_text'].replace('50.00', '70.00')
    assert split.task_identity(alias) == split.task_identity(first)
    with pytest.raises(RuntimeError, match='leakage'):
        split.assert_allowed(alias, True)
    assert split.assert_allowed(alias, False) == split.task_identity(first)
    changed = dict(alias, instruction_text='A genuinely different requested objective')
    assert split.task_identity(changed) != split.task_identity(first)


def test_catalog_drift_rejected_before_policy_interaction(synthetic_split):
    split, goals, _ = synthetic_split
    with pytest.raises(RuntimeError, match='catalog differs'):
        split.partition_goals(goals[:-1], True)
    changed = copy.deepcopy(goals)
    changed[0]['attributes'].append('new-constraint')
    with pytest.raises(RuntimeError, match='catalog differs'):
        split.partition_goals(changed, False)


def test_wrong_actual_task_binding_rejected_before_generation():
    from fixed_validation import reset_fixed_validation, manifest
    env = SimpleNamespace(_rng=np.random.RandomState(1000), goal_idxs=range(140),
        env_num=140, group_n=1, is_train=False)
    previous = env._rng
    def wrong():
        return ['observation']*140, [{'task_id':'wrong'}]*140
    with pytest.raises(RuntimeError, match='Actual validation task identities'):
        reset_fixed_validation(env, wrong, 1000)
    assert env._rng is previous


@pytest.mark.parametrize('is_train', [True, False])
def test_worker_filters_constructor_and_automatic_resets(synthetic_split, is_train):
    split, goals, value = synthetic_split
    class Native:
        def __init__(self, seed, kwargs):
            self.env = SimpleNamespace(server=SimpleNamespace(goals=copy.deepcopy(goals), user_sessions={}))
            self.env.unwrapped = self.env
            self.env.session = 'session'
            def reset(session=0):
                self.env.server.user_sessions['session'] = {'goal': self.env.server.goals[session]}
                return 'observation', None
            self.env.reset = reset
            reset()
        def reset(self, index):
            observation, _ = self.env.reset(index)
            return observation, {}
        def step(self, action):
            self.env.reset(len(self.env.server.goals)-1)  # Simulates native terminal auto-reset.
            return 'terminal observation', 10., True, {'won':True}
    source = ROOT/'src/webshop_worker.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    scope = {'NativeWebshopWorker':Native, 'np':np, 'manifest':split.manifest,
        'partition_goals':split.partition_goals, 'assert_allowed':split.assert_allowed,
        'task_identity':split.task_identity}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),scope)
    worker = scope['DisjointWebshopWorker'](0, {}, is_train)
    audit = worker.get_split_audit()
    assert audit['opposite_split_overlap'] == 0
    assert audit['goal_entries'] == (len(goals)-140 if is_train else 140)
    _, info = worker.reset(0)
    before = info['task_id']
    _, reward, done, info = worker.step('click[buy now]')
    assert reward == 10. and done and info['task_id'] == before
    assert worker._current_id() == split.task_identity(worker.env.server.goals[-1])
    with pytest.raises(RuntimeError, match='outside this split'):
        worker.reset(len(worker.env.server.goals))


@pytest.mark.parametrize('raw,done,expected', [(1.,True,10.),(.8,True,0.),(1.,False,0.),(0.,False,0.)])
def test_original_webshop_reward_not_changed(raw, done, expected):
    path = UPSTREAM/'agent_system/environments/env_package/webshop/envs.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'WebshopWorker')
    space = {}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),space)
    worker = space['WebshopWorker'].__new__(space['WebshopWorker'])
    worker.env = SimpleNamespace(step=lambda a:('observation',raw,done,{}),
        get_available_actions=lambda:{'clickables':['buy now']})
    obs,reward,ended,info = worker.step('click[buy now]')
    assert reward == expected and ended == done and info['task_score'] == raw
    assert info['won'] == (expected == 10.)


def test_download_checksums_and_does_not_replace_existing_file(tmp_path):
    from assets import download_file
    source = tmp_path/'public.json'
    source.write_bytes(b'public fixture')
    target = tmp_path/'download.json'
    expected = {'bytes':source.stat().st_size, 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    download_file(target,[source.as_uri()],expected)
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b'different')
    with pytest.raises(RuntimeError,match='Existing asset differs'):
        download_file(target,[source.as_uri()],expected)
    assert target.read_bytes() == b'different'


def test_dry_run_and_dummy_parquet_need_no_assets_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'src/run.py'), 'train','--size','1.5b',
        '--run-dir',str(tmp_path/'unused'),'--dry-run'],text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'unused').exists()
    assert 'adv_estimator: grpo' in result.stdout and 'env_name: Webshop' in result.stdout
    assert 'gate_beta' not in result.stdout
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    from run import data
    data(tmp_path)
    assert len(pd.read_parquet(tmp_path/'train.parquet')) == 16
    assert len(pd.read_parquet(tmp_path/'val.parquet')) == 140


def test_release_allowlist_and_history_privacy():
    result = subprocess.run([sys.executable,str(ROOT/'scripts/check_release.py')],text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
