"""Device selection shared by training, map loading and result loading."""


def resolve_device(device="auto", *, gpu_index=0):
    """Resolve auto, cpu, mps, cuda or cuda:N without changing training settings.

    Auto preserves the engine's CUDA, MPS, CPU priority. Explicit accelerator
    requests fail if unavailable; they are never silently moved to the CPU.
    CUDA indices refer to the devices visible to the current process.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available() and gpu_index >= 0:
            return resolve_device(f"cuda:{gpu_index}")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N'.")
    try:
        selected = torch.device(device)
    except (ValueError, RuntimeError) as error:
        raise ValueError(
            f"Invalid device {device!r}. Use 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N'."
        ) from error
    if selected.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device {device!r}. Use CPU, MPS or CUDA.")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {str(selected)!r}, but CUDA is unavailable in this Python environment. "
                "Check your PyTorch installation and NVIDIA driver, or choose device='cpu'."
            )
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {str(selected)!r}, but only {torch.cuda.device_count()} CUDA "
                "device(s) are visible. Choose a visible index starting at zero."
            )
    elif selected.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "Requested 'mps', but MPS is unavailable in this Python environment. "
                "Use a compatible Mac and PyTorch build, or choose device='cpu'."
            )
    return selected
