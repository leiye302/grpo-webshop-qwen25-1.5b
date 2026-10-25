"""Report actual native WebShop validation success and continuous task score."""
import argparse
import csv
import json
from pathlib import Path
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    rows = []
    task_identities = set()
    for size in ('1.5b', '3b'):
        run = args.runs / ('grpo_' + size)
        for path in sorted((run / 'evaluation').glob('update_*/summary.json')):
            item = json.loads(path.read_text())
            identity = item.get('fixed_validation_identity')
            if not identity:
                raise RuntimeError('Random/legacy validation results cannot enter the fixed-task report')
            task_identities.add((identity['manifest_sha256'], identity['selection_sha256'],
                                 identity['initial_observations_sha256']))
            rows.append({'model': size, **{k: item[k] for k in (
                'update', 'successes', 'episodes', 'success_rate', 'task_score',
                'task_score_0_to_100', 'selection_sha256', 'reasoning_open_rate',
                'strict_protocol_rate', 'native_action_valid_rate')},
                'validation_manifest_sha256': identity['manifest_sha256'],
                'initial_observations_sha256': identity['initial_observations_sha256']})
    if not rows:
        raise RuntimeError('No real validation results exist yet')
    if len(task_identities) != 1:
        raise RuntimeError('Validation task identities differ; refusing an incomparable combined curve')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'webshop.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Wrote webshop.csv; install matplotlib for curves.')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for size in ('1.5b', '3b'):
        selected = sorted((r for r in rows if r['model'] == size), key=lambda x:x['update'])
        if selected:
            for ax, key in zip(axes, ('success_rate','task_score')):
                ax.plot([r['update'] for r in selected], [100*r[key] for r in selected],
                        marker='o', label='GRPO Qwen2.5-' + size)
    for ax, label in zip(axes, ('Native success (%)','Native task score (0-100)')):
        ax.set_xlabel('Training update')
        ax.set_ylabel(label)
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'webshop.png', dpi=180)
    print('Wrote webshop.csv and webshop.png; no missing results imputed.')


if __name__ == '__main__':
    main()
