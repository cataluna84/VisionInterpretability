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

## Optimization Modes

### `baseline` (Default)
Standard PyTorch eager execution. Use for establishing baseline metrics.

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
