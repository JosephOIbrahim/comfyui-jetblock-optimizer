"""
JetBlock Optimizer v4.0 - ComfyUI Node Registration

Only v4 nodes are exposed. Legacy v2 nodes have been removed.
"""

from .nodes_v4 import (
    V4_NODE_CLASS_MAPPINGS,
    V4_NODE_DISPLAY_NAME_MAPPINGS
)

# Export only v4 nodes
NODE_CLASS_MAPPINGS = V4_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = V4_NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
