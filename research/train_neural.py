"""Optional FP32 MLP fit on cached weak labels; does not estimate generalization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from devices import get_universal_device, save_weights, load_weights


class NeuralCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_scale', torch.ones(input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net((x - self.feature_mean) / self.feature_scale).squeeze(-1)


def load_neural_critic(weights, report_path, device=None):
    """Respect the saved checkpoint's own feature width, including the old baseline."""
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    names = report['feature_names']
    model = load_weights(NeuralCritic(len(names)), weights, device)
    model.feature_names = names
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='artifacts/training_v4/training_features.npz')
    parser.add_argument('--output', type=Path, default=Path('outputs/neural'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=600)
    args = parser.parse_args()
    with np.load(args.features, allow_pickle=False) as data:
        x, y, names = data['x'], data['y'], data['feature_names'].tolist()
    torch.manual_seed(42)
    device = get_universal_device(args.device)
    model = NeuralCritic(x.shape[1]).float()
    model.feature_mean.copy_(torch.tensor(x.mean(0)))
    model.feature_scale.copy_(torch.tensor(np.maximum(x.std(0), .01)))
    model.to(device)
    xx = torch.tensor(x, dtype=torch.float32, device=device)
    yy = torch.tensor(y / 100, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = ((model(xx) - yy) ** 2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'epoch={epoch} loss={float(loss.detach().cpu()):.6f}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    save_weights(model, args.output / 'critic_mlp.safetensors')
    report = {'device': str(device), 'precision': 'FP32', 'epochs': args.epochs,
              'feature_names': names, 'validation': 'Fit only; no independent validation performed.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
