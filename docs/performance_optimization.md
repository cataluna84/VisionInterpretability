# Performance Optimization Guide

This document outlines optimization strategies for the activation spectrum pipeline.

## Research Summary

| Optimization | Expected Speedup | Complexity | Notes |
|--------------|-----------------|------------|-------|
| `torch.compile` | 2-5x | Low | JIT compilation via Triton backend |
| `pin_memory=True` | 1.2-1.5x | Low | Faster CPU→GPU transfers |
| `non_blocking=True` | 1.1-1.3x | Low | Async GPU transfers |
| DataLoader workers | 1.5-3x | Low | Parallel data loading |
| Mixed precision (FP16) | 1.5-2x | Medium | Requires `torch.amp` |
| Local WDS shards | 5-10x vs HTTPS | Low | Used by Segment 3 Canonical |

## Optimization Modes

### `baseline` (Default)
Standard PyTorch eager execution. Use for establishing baseline metrics.

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### `compile`
```python
model = torch.compile(model, mode="reduce-overhead")
```
JIT compiles the model using TorchInductor/Triton. Best for repeated inference.

### `optimized`
```python
model = torch.compile(model, mode="max-autotune")
tensors = tensors.to(device, non_blocking=True)
```
Full optimization with async transfers and maximum autotuning.

## Data Loading Strategy

### Local WDS Shards (Recommended)

The Segment 3 Canonical notebook uses **local WebDataset shards** stored in
`data/imagenet-1k-wds/` (~144 GB). This avoids the bottleneck
of HTTPS streaming and enables `num_workers > 0` for parallel I/O.

| Approach | Throughput | Bottleneck |
|----------|------------|------------|
| HTTPS streaming (urllib) | Slow | Network + GIL |
| HuggingFace `datasets` streaming | Moderate | Buffer management |
| **Local WDS shards + DataLoader** | **Fast** | **Disk I/O (SSD ideal)** |

Download the shards on first run:
```bash
uv run jupyter lab notebooks/Segment_3_canonical_Windows.ipynb
```
The notebook auto-downloads via `huggingface_hub` with resume support.

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| Total Samples | Number of images processed |
| Total Time | Wall-clock time in seconds |
| Throughput | Images per second |
| GPU Utilization | % of GPU compute used |
| Memory Peak | Maximum GPU memory allocated |

## References

- [PyTorch Compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Triton Documentation](https://triton-lang.org/)
- [HuggingFace Datasets Streaming](https://huggingface.co/docs/datasets/stream)
- [WebDataset Documentation](https://webdataset.github.io/webdataset/)
- [HuggingFace Hub Download](https://huggingface.co/docs/huggingface_hub/guides/download)
