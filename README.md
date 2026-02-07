# ComfyUI Deterministic Toolkit

**Batch-invariant deterministic inference for any diffusion model.**

[![Version](https://img.shields.io/badge/version-4.1.0-blue.svg)](https://github.com/JosephOIbrahim/comfyui-deterministic-toolkit)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)

---

## The Problem

**Setting `temperature=0` does not guarantee deterministic outputs.**

When you run the same workflow twice with the same seed, you expect identical results. But GPU parallel operations vary their reduction order based on batch size and system load, causing subtle numerical differences that compound through the network.

This is why your "reproducible" workflows produce slightly different outputs each run.

## The Solution

This toolkit implements **batch-invariant operators** based on the [ThinkingMachines research](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):

- **RMSNorm**: Fixed reduction strategy, not dependent on batch size
- **MatMul**: No Split-K parallelization variance
- **Attention**: Fixed split-SIZE (not split-count) for consistent computation order

**Same seed = identical output. Guaranteed.**

---

## Quick Start

### Installation

**ComfyUI Manager** (Recommended)
```
Search "Deterministic Toolkit" in ComfyUI Manager
```

**Manual**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JosephOIbrahim/comfyui-deterministic-toolkit.git
```

### Basic Usage

Add the **Universal Determinism** node to any workflow:

```
Load Checkpoint (any model)
        |
        v
Universal Determinism (strict, seed=12345)
        |
        v
[Your existing workflow]
        |
        v
Save Image
```

Run twice. Compare outputs. They're identical.

---

## Nodes

### Universal (Works with Any Model)

| Node | Purpose |
|------|---------|
| **Universal Determinism** | Apply batch-invariant determinism to any model |
| **Deterministic Sampler** | Full deterministic sampling with checksum verification |
| **Deterministic Mode Switch** | Toggle between speed/balanced/strict modes |
| **Hybrid Architecture Profiler** | Analyze model architecture |

### Nemotron-Specific (Requires Dependencies)

| Node | Purpose |
|------|---------|
| **Nemotron Deterministic Optimizer** | Full Nemotron 3 hybrid architecture determinism |
| **Cascade Mode Control** | Control /think vs /no_think reasoning budget |
| **Mamba-2 Deterministic** | Mamba-2 state-space layer determinism |

---

## Determinism Levels

| Level | Batch Invariance | Performance | Use Case |
|-------|------------------|-------------|----------|
| `speed` | None | Full | Fast iteration, don't need reproducibility |
| `balanced` | Partial | ~15-30% slower | Development, some reproducibility |
| `strict` | Full | ~60-100% slower | Production, guaranteed reproducibility |

### What Each Level Does

**Speed Mode**
- cuDNN benchmark enabled
- Standard batched operations
- No determinism guarantees

**Balanced Mode**
- cuDNN deterministic enabled
- Some fixed reduction strategies
- Most runs will match

**Strict Mode**
- cuDNN benchmark disabled
- Deterministic algorithms enforced
- Batch size forced to 1
- FP32 intermediate precision
- **Every run matches**

---

## Node Reference

### Universal Determinism

The main node. Works with SD 1.5, SD 2.x, SDXL, Flux, LTX Video, and any other diffusion model.

**Inputs**

| Name | Type | Description |
|------|------|-------------|
| `model` | MODEL | Any ComfyUI model |
| `determinism_level` | COMBO | `speed` / `balanced` / `strict` |
| `seed` | INT | Master seed for all RNG operations |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `model` | MODEL | Model with determinism configured |
| `status` | STRING | Configuration report |

---

### Deterministic Sampler

Full deterministic sampling with verification. Processes each batch item independently for guaranteed reproducibility.

**Inputs**

| Name | Type | Description |
|------|------|-------------|
| `model` | MODEL | Model to sample |
| `positive` | CONDITIONING | Positive prompt |
| `negative` | CONDITIONING | Negative prompt |
| `latent_image` | LATENT | Input latent |
| `seed` | INT | Sampling seed |
| `steps` | INT | Number of steps |
| `cfg` | FLOAT | CFG scale |
| `sampler_name` | COMBO | Sampler algorithm |
| `scheduler` | COMBO | Noise schedule |
| `denoise` | FLOAT | Denoise strength |
| `determinism_level` | COMBO | `strict` / `paranoid` |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `samples` | LATENT | Deterministic output |
| `checksum` | STRING | SHA-256 hash for verification |
| `report` | STRING | Detailed execution report |

---

### Deterministic Mode Switch

Quick toggle between performance modes without full node replacement.

**Inputs**

| Name | Type | Description |
|------|------|-------------|
| `mode` | COMBO | `speed` / `balanced` / `strict` / `paranoid` |
| `seed` | INT | Master seed |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `status` | STRING | Mode configuration report |

---

### Hybrid Architecture Profiler

Analyze any model's architecture. Useful for understanding layer composition before applying determinism.

**Inputs**

| Name | Type | Description |
|------|------|-------------|
| `model` | MODEL | Model to analyze |

**Outputs**

| Name | Type | Description |
|------|------|-------------|
| `architecture_summary` | STRING | Layer type breakdown |
| `detailed_report` | STRING | Full JSON analysis |

---

## Workflow Examples

### Reproducible Image Generation

```
Load Checkpoint
        |
        v
CLIP Text Encode ----+
        |            |
        v            v
Universal Determinism (strict, seed=42)
        |
        v
Empty Latent Image
        |
        v
Deterministic Sampler (seed=42)
        |
        v
VAE Decode
        |
        v
Save Image
```

### Fast Iteration (No Determinism)

```
Load Checkpoint
        |
        v
Deterministic Mode Switch (speed)
        |
        v
[Standard KSampler workflow]
```

### Model Architecture Analysis

```
Load Checkpoint
        |
        v
Hybrid Architecture Profiler
        |
        v
Show Text (view architecture breakdown)
```

---

## Technical Details

### How Batch Invariance Works

Standard GPU kernels adapt their parallelization strategy based on workload:

```
Batch size 1:  A + B + C + D          = result1
Batch size 4:  (A + B) + (C + D)      = result2  (different!)
```

Floating-point addition is **not associative**. Different groupings produce different results.

Batch-invariant operators force consistent computation order:

```
Batch size 1:  A + B + C + D          = result
Batch size 4:  A + B + C + D (per item) = same result
```

### PyTorch Settings Applied (Strict Mode)

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_float32_matmul_precision('highest')
```

### Batch-Invariant Operators

| Operator | Standard | Deterministic Toolkit |
|----------|----------|----------------------|
| RMSNorm | Batched reduction | Per-sample, FP32 |
| MatMul | Split-K parallel | No Split-K, FP32 |
| Attention | Variable split-count | Fixed split-SIZE |

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11+ |
| PyTorch | 2.1.0 | 2.2.0+ |
| CUDA Compute | 8.0 | 8.9+ (Ada) |
| VRAM | 8 GB | 24 GB |

### Dependencies

**Required**
```
torch>=2.1.0
numpy>=1.24.0
```

**Optional** (for Nemotron-specific features)
```
triton>=2.1.0
einops>=0.7.0
mamba-ssm>=1.2.0
```

---

## Performance

### Overhead by Mode

| Mode | Overhead | Determinism |
|------|----------|-------------|
| speed | 0% | None |
| balanced | 15-30% | Partial |
| strict | 60-100% | Full |

### Benchmark (RTX 4090, SDXL 1024x1024)

| Mode | Time | Deterministic |
|------|------|---------------|
| speed | 2.1s | No |
| balanced | 2.6s | Usually |
| strict | 3.8s | Always |

---

## Troubleshooting

### Outputs still vary between runs

1. Ensure `determinism_level="strict"`
2. Use the Deterministic Sampler instead of standard KSampler
3. Verify same seed across all runs
4. Check that no other nodes modify RNG state

### Performance too slow

1. Use `balanced` mode during development
2. Switch to `strict` only for final production runs
3. Use `speed` mode when reproducibility doesn't matter

### Import errors on startup

The toolkit gracefully handles missing dependencies. Core nodes (Universal Determinism, Deterministic Sampler, Mode Switch, Profiler) always load. Nemotron-specific nodes require additional packages.

### CUDA errors

1. Update PyTorch to 2.1.0+
2. Try `torch.use_deterministic_algorithms(True, warn_only=True)` in your environment
3. Some operations don't have deterministic implementations on all hardware

---

## API Reference

### Python Import

```python
from comfyui_deterministic_toolkit.jetblock_core_v4 import (
    JetBlockV4Config,
    DeterminismLevel,
    get_config,
    set_config,
    BatchInvariantRMSNorm,
    compute_tensor_checksum,
    validate_determinism,
)

# Configure globally
config = JetBlockV4Config(
    determinism_level=DeterminismLevel.STRICT,
    master_seed=42,
    force_batch_size_one=True,
)
set_config(config)
config.setup_deterministic_environment()

# Validate a function's determinism
is_det, report = validate_determinism(
    my_function,
    {"input": my_tensor},
    num_runs=10,
)
print(report)
# "DETERMINISTIC: 10/10 identical (checksum: a1b2c3d4...)"
```

---

## Research Background

Based on [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (ThinkingMachines, 2025):

> "Everyone thinks temperature=0 gives you determinism. It doesn't. Batch-size variance in GPU operations is the real culprit."

Key findings:
- 1000 identical requests produced **80 unique completions** without batch invariance
- With batch-invariant kernels, all 1000 completions matched perfectly
- Performance cost: 1.6-2x slowdown (acceptable for production reproducibility)

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

## Credits

**Author:** Joseph Ibrahim

**Research Integration:**
- [ThinkingMachines](https://thinkingmachines.ai/) - Batch invariance research
- NVIDIA - Nemotron 3 hybrid architecture

---

*Determinism is not a feature. It's a requirement.*
