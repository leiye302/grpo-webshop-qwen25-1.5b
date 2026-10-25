"""Fixed native first-validation task bindings; never applied to training envs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from common import ROOT, atomic_json, load, sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def manifest():
    value = load('fixed_validation128.json')
    indices = np.random.RandomState(1000).choice(500, 128, replace=False).tolist()
    expected = [{'worker': i, 'worker_seed': 1000+i, 'goal_index': index}
                for i, index in enumerate(indices)]
    if (value['schema'] != 1 or value['count'] != 128 or value['base_seed'] != 1000
            or value['selection'] != expected
            or value['upstream_commit'] != load('upstream.json')['commit']
            or value['assets_lock_sha256'] != sha(ROOT / 'configs/assets.json')):
        raise RuntimeError('Fixed WebShop validation manifest/configuration differs')
    return value


def native_validation_selection(env, base_seed):
    """Preview original RandomState.choice, without advancing its real state."""
    preview = np.random.RandomState()
    preview.set_state(env._rng.get_state())
    indices = np.repeat(preview.choice(env.goal_idxs, size=env.env_num, replace=False), env.group_n).tolist()
    return [{'worker': i, 'worker_seed': base_seed + i // env.group_n, 'goal_index': int(index)}
            for i, index in enumerate(indices)]


def reset_fixed_validation(env, original_reset, base_seed):
    """Use the original env.reset, temporarily selecting the manifest's first draw."""
    fixed = manifest()
    if (getattr(env, 'is_train', None) is not False or env.env_num != fixed['count']
            or env.group_n != 1 or base_seed != fixed['base_seed']
            or list(env.goal_idxs) != list(range(500))):
        raise RuntimeError('Fixed evaluation requires the native 128-worker validation environment')
    previous = env._rng.get_state()
    try:
        env._rng.seed(fixed['base_seed'])
        if native_validation_selection(env, base_seed) != fixed['selection']:
            raise RuntimeError('Native sampler does not match the fixed manifest')
        result = original_reset()
    finally:
        env._rng.set_state(previous)
    observations, infos = result
    if len(observations) != fixed['count'] or len(infos) != fixed['count']:
        raise RuntimeError('Missing fixed validation observations')
    identity = {
        'protocol': fixed['protocol'],
        'manifest_sha256': sha(ROOT / 'configs/fixed_validation128.json'),
        'selection_sha256': digest(fixed['selection']),
        'initial_observations_sha256': digest(list(observations)),
    }
    return result, identity


def record_identity(run_dir, identity):
    path = Path(run_dir) / 'fixed_validation_identity.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Fixed validation initial tasks changed; refusing incomparable evaluation')
    else:
        atomic_json(path, identity)
