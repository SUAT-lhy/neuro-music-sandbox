"""Refit the CPU critic from retained features; this is not fresh validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import sklearn
from critic_v4 import VoiceCritic, V4_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path,
                        default=Path('artifacts/training_v4/training_features.npz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/training'))
    args = parser.parse_args()
    with np.load(args.features, allow_pickle=False) as cache:
        if cache['feature_names'].tolist() != V4_FEATURE_NAMES:
            raise ValueError('Feature schema mismatch')
        x, y = cache['x'], cache['y']
        hx, hy, hw = cache['human_x'], cache['human_y'], cache['human_weights']
    args.output.mkdir(parents=True, exist_ok=True)
    preference = VoiceCritic(42)
    preference.model.fit(np.concatenate([x, hx]), np.r_[y, hy],
                         sample_weight=np.r_[np.ones(len(y)), hw])
    preference.is_trained = True
    preference.save(args.output / 'critic_model.pkl')
    technical = VoiceCritic(42)
    technical.model.fit(x, np.where(y == 55, 90.0, y))
    technical.is_trained = True
    technical.save(args.output / 'technical_model.pkl')
    report = {'weak_rows': len(y), 'human_rows': len(hy),
              'sklearn': sklearn.__version__, 'feature_names': V4_FEATURE_NAMES,
              'validation': 'No new validation: cached features lack work-group IDs. '
                            'Historical grouped/leave-one-rating-out results are retained separately.'}
    (args.output / 'refit_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'weak_rows': len(y), 'human_rows': len(hy)}))


if __name__ == '__main__':
    main()
