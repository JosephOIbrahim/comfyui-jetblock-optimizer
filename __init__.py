"""
ComfyUI JetBlock Optimizer v4.0
===============================

Batch-invariant determinism for any ComfyUI model.

Based on ThinkingMachines research: batch-size variance in GPU kernels
causes non-determinism, not temperature settings. JetBlock fixes this
at the tensor operation level.

Supports:
- Universal determinism (any model: SD, SDXL, Flux, LTX Video)
- Nemotron 3 hybrid architecture optimization (Mamba-2 + MoE + Attention)
- Cascade mode control (/think vs /no_think)

Optimized for RTX 4090 with 24GB VRAM

References:
- ThinkingMachines "Defeating Nondeterminism in LLM Inference" (2025)
- Nemotron 3 Technical Report
"""

__version__ = "4.0.0"

# Import v4 nodes directly (nodes.py is kept for backwards compat only)
try:
    from .nodes_v4 import V4_NODE_CLASS_MAPPINGS, V4_NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS = dict(V4_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS = dict(V4_NODE_DISPLAY_NAME_MAPPINGS)
    V4_LOADED = True
except ImportError as e:
    print(f"[JetBlock] Warning: v4 nodes failed to load: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    V4_LOADED = False

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

WEB_DIRECTORY = "./js"

# Startup message
print("=" * 60)
print(f"JetBlock Optimizer v{__version__} loaded")
if V4_LOADED:
    print(f"  {len(NODE_CLASS_MAPPINGS)} nodes registered")
    for name in sorted(NODE_CLASS_MAPPINGS.keys()):
        print(f"    - {name}")
else:
    print("  v4 nodes UNAVAILABLE (import error)")
    print("  No nodes registered")
print("=" * 60)
