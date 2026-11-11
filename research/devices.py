"""Portable FP32 devices and CPU-only tensor checkpoints. No vendor torch replacement."""
from pathlib import Path


def get_universal_device(requested='auto'):
    import torch
    musa_error = None
    try:
        import torch_musa  # noqa: F401 -- registers the MUSA backend
    except ImportError as exc:
        musa_error = str(exc)
    if requested != 'auto':
        device = torch.device(requested)
        if device.type == 'musa' and (not hasattr(torch,'musa') or not torch.musa.is_available()):
            raise RuntimeError(f'MUSA explicitly requested but unavailable: {musa_error}')
        return device
    if hasattr(torch,'musa') and torch.musa.is_available():
        return torch.device('musa')
    if torch.cuda.is_available():
        return torch.device('cuda')  # NVIDIA CUDA and AMD ROCm both expose this interface
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_weights(model,path):
    from safetensors.torch import save_file
    weights = {k:v.detach().to('cpu').contiguous().clone() for k,v in model.state_dict().items()}
    save_file(weights,str(path),metadata={'precision':'float32','format':'pt'})


def load_weights(model,path,device=None):
    import torch
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(str(path),device='cpu')
    else:
        weights = torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True)
    return model.to(device if device is not None else get_universal_device())
