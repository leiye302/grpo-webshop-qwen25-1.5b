"""Real CPU WebShop sanity: search, select, customize and buy; no Actor scores."""
import argparse
import os
from pathlib import Path
import sys
from common import ROOT, UPSTREAM, WEBSHOP, atomic_json
from assets import verify
from setup import verify as verify_source
from fixed_validation import manifest, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/environment_smoke.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    verify_source()
    receipt = verify()
    sys.path[:0] = [str(UPSTREAM), str(WEBSHOP)]
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    records = []
    fixed = manifest()['selection']
    cases = [(0, 0)] + [(r['worker_seed'], r['goal_index']) for r in (fixed[0], fixed[-1])]
    for seed, goal_index in cases:
        worker = WebshopWorker(seed, {'observation_mode': 'text', 'num_products': None, 'human_goals': False,
            'file_path': str(WEBSHOP / 'data/items_shuffle_1000.json'),
            'attr_path': str(WEBSHOP / 'data/items_ins_v2_1000.json')})
        try:
            initial, _ = worker.reset(goal_index)
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
                            'repeat_reset_identical': True, 'initial_observation_sha256': digest(initial),
                            'actions': actions, 'native_reward': reward, 'task_score': info['task_score']})
        finally:
            worker.close()
    atomic_json(args.out, {'status': 'passed', 'cpu_environment_only': True,
        'not_actor_or_benchmark_performance': True, 'corpus_sha256': receipt['corpus_sha256'], 'cases': records})
    print('Passed real native WebShop CPU environment smoke; not model evaluation.')


if __name__ == '__main__':
    main()
