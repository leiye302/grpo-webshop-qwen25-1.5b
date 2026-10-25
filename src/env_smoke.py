"""Real CPU WebShop sanity: search, select, customize and buy; no Actor scores."""
import argparse
import os
from pathlib import Path
import sys
from common import ROOT, UPSTREAM, WEBSHOP, atomic_json
from assets import verify
from setup import verify as verify_source
from fixed_validation import manifest, digest
from task_split import make_manifest, manifest as split_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/environment_smoke.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    verify_source()
    receipt = verify()
    sys.path[:0] = [str(UPSTREAM), str(WEBSHOP)]
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from webshop_worker import DisjointWebshopWorker
    kwargs = {'observation_mode': 'text', 'num_products': None, 'human_goals': False,
        'file_path': str(WEBSHOP / 'data/items_shuffle_1000.json'),
        'attr_path': str(WEBSHOP / 'data/items_ins_v2_1000.json')}
    # Rebuild the published split with the real, unmodified native goal generator.
    canonical = WebshopWorker(1000, dict(kwargs))
    try:
        if make_manifest(canonical.env.unwrapped.server.goals) != split_manifest():
            raise RuntimeError('Public140-task split does not reproduce from the native catalog')
    finally:
        canonical.close()
    records = []
    fixed = manifest()['selection']
    cases = [(0, True, 0), (15, True, 1)] + [
        (r['worker_seed'], False, r['goal_index']) for r in (fixed[0], fixed[-1])]
    for seed, is_train, goal_index in cases:
        worker = DisjointWebshopWorker(seed, kwargs, is_train)
        try:
            audit = worker.get_split_audit()
            if audit['opposite_split_overlap']:
                raise RuntimeError('Training and validation task pools overlap')
            initial, initial_info = worker.reset(goal_index)
            if not is_train and initial_info['task_id'] != fixed[goal_index]['task_id']:
                raise RuntimeError('Actual held-out goal differs from the public140 manifest')
            if is_train:
                # Training retains all other native entries, rather than a deduplicated pool.
                if audit['goal_entries'] != split_manifest()['training_goal_entries']:
                    raise RuntimeError('Wrong filtered training pool size')
                records.append(dict(audit, reset_task_id=initial_info['task_id']))
                continue
            env = worker.env.unwrapped
            goal = env.server.user_sessions[env.session]['goal']
            product = env.server.product_item_dict[goal['asin']]
            actions = ['search[<q> ' + product['query'] + ']']
            worker.step(actions[-1])
            for _ in range(100):
                available = worker.get_available_actions()['clickables']
                if goal['asin'].lower() in available:
                    break
                if 'next >' not in available:
                    raise RuntimeError('Target was not reachable through native search')
                actions.append('click[next >]')
                worker.step(actions[-1])
            actions.append('click[' + goal['asin'] + ']')
            worker.step(actions[-1])
            for value in goal['goal_options'].values():
                actions.append('click[' + value + ']')
                worker.step(actions[-1])
            actions.append('click[buy now]')
            _, reward, done, info = worker.step(actions[-1])
            if not done or reward != 10.0 or not info['won'] or info['task_score'] != 1.0:
                raise RuntimeError('Native exact-goal purchase did not yield the original sparse success reward')
            repeated, _ = worker.reset(goal_index)
            if repeated != initial:
                raise RuntimeError('Native fixed-goal reset changed the initial task after purchase')
            records.append({'seed': seed, 'goal_index': goal_index, 'goal_count': len(env.server.goals),
                            'task_id': initial_info['task_id'], 'split_audit': audit,
                            'repeat_reset_identical': True, 'initial_observation_sha256': digest(initial),
                            'actions': actions, 'native_reward': reward, 'task_score': info['task_score']})
        finally:
            worker.close()
    atomic_json(args.out, {'status': 'passed', 'cpu_environment_only': True,
        'not_actor_or_benchmark_performance': True, 'corpus_sha256': receipt['corpus_sha256'],
        'fixed140_manifest_rebuilt_from_native': True, 'diversity': split_manifest()['diversity'], 'cases': records})
    print('Passed real native WebShop CPU environment smoke; not model evaluation.')


if __name__ == '__main__':
    main()
