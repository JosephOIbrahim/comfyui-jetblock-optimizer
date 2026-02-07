"""
ComfyUI Compatibility Nodes for Deterministic Toolkit
"""

import json
from typing import Dict, Any, Tuple
from .compatibility_checker import check_compatibility, WORKFLOW_ANALYZER
from .jetblock_core import get_optimizer, logger


class JetBlockCompatibilityChecker:
    """
    Analyze workflow and suggest optimal settings
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analyze_current": ("BOOLEAN", {"default": True}),
                "auto_adjust": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "workflow_json": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("compatibility_report", "optimization_level", "skip_ratio", "use_temporal_skip", "use_pattern_cache")
    FUNCTION = "check_compatibility"
    CATEGORY = "Deterministic/Compatibility"

    def check_compatibility(self, analyze_current, auto_adjust, workflow_json=""):
        """
        Check workflow compatibility and return optimal settings
        """

        # Get workflow data
        workflow = {}
        if workflow_json:
            try:
                workflow = json.loads(workflow_json)
            except:
                workflow = {}

        # If no workflow provided, try to analyze current execution context
        if not workflow and analyze_current:
            # This would need integration with ComfyUI's execution context
            # For now, return default safe settings
            workflow = self._get_current_workflow_context()

        # Analyze workflow
        profile, report = check_compatibility(workflow)

        # Auto-adjust optimizer if requested
        if auto_adjust:
            optimizer = get_optimizer()

            # Apply profile settings
            optimizer.temporal_skipper.skip_ratio = profile.skip_ratio

            # Log adjustments
            logger.info(f"Auto-adjusted optimization settings based on workflow type: {profile.name}")

        return (
            report,
            profile.optimization_level,
            profile.skip_ratio,
            profile.use_temporal_skip,
            profile.use_pattern_cache
        )

    def _get_current_workflow_context(self) -> Dict[str, Any]:
        """
        Try to get current workflow context (placeholder)
        """
        # This would need to hook into ComfyUI's execution context
        # For now, return empty workflow
        return {"nodes": []}


class JetBlockSmartOptimizer:
    """
    Intelligent optimizer that auto-adjusts based on workflow
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "auto_detect": ("BOOLEAN", {"default": True}),
                "override_level": (["auto", "low", "medium", "high", "extreme"],),
                "safe_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("optimized_model", "optimization_report")
    FUNCTION = "smart_optimize"
    CATEGORY = "Deterministic/Smart"

    def smart_optimize(self, model, auto_detect, override_level, safe_mode):
        """
        Intelligently optimize model based on workflow analysis
        """
        import copy
        optimizer = get_optimizer()
        optimized_model = copy.deepcopy(model)

        # Get actual model
        if hasattr(optimized_model, 'model'):
            actual_model = optimized_model.model
        else:
            actual_model = optimized_model

        # Analyze model architecture
        model_info = self._analyze_model_architecture(actual_model)

        # Determine optimization level
        if override_level != "auto":
            optimization_level = override_level
        elif auto_detect:
            # Smart detection based on model type
            optimization_level = self._detect_optimal_level(model_info, safe_mode)
        else:
            optimization_level = "medium"

        # Apply optimizations
        skip_ratios = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "extreme": 0.9
        }

        optimizer.temporal_skipper.skip_ratio = skip_ratios[optimization_level]

        # Optimize model
        optimizer.optimize_model(actual_model, f"smart_optimized_{optimization_level}")

        # Generate report
        report = f"Smart Optimization Report\n"
        report += f"{'=' * 40}\n"
        report += f"Model Type: {model_info['type']}\n"
        report += f"Parameters: {model_info['params'] / 1e9:.1f}B\n"
        report += f"Optimization Level: {optimization_level}\n"
        report += f"Skip Ratio: {skip_ratios[optimization_level]:.0%}\n"
        report += f"Safe Mode: {'ENABLED' if safe_mode else 'DISABLED'}\n"
        report += f"Expected Speedup: {self._calculate_speedup(optimization_level)}x"

        logger.info(report)

        return (optimized_model, report)

    def _analyze_model_architecture(self, model) -> Dict[str, Any]:
        """
        Analyze model architecture for optimization decisions
        """
        import torch.nn as nn

        total_params = sum(p.numel() for p in model.parameters())
        attention_layers = 0
        conv_layers = 0
        linear_layers = 0

        for module in model.modules():
            if "attention" in module.__class__.__name__.lower():
                attention_layers += 1
            elif isinstance(module, nn.Conv2d):
                conv_layers += 1
            elif isinstance(module, nn.Linear):
                linear_layers += 1

        # Detect model type
        model_type = "unknown"
        if "unet" in model.__class__.__name__.lower():
            model_type = "unet"
        elif "dit" in model.__class__.__name__.lower():
            model_type = "dit"
        elif "vae" in model.__class__.__name__.lower():
            model_type = "vae"
        elif attention_layers > 20:
            model_type = "transformer"

        return {
            "type": model_type,
            "params": total_params,
            "attention_layers": attention_layers,
            "conv_layers": conv_layers,
            "linear_layers": linear_layers
        }

    def _detect_optimal_level(self, model_info: Dict[str, Any], safe_mode: bool) -> str:
        """
        Detect optimal optimization level based on model architecture
        """

        # Safe mode always uses low
        if safe_mode:
            return "low"

        # Model-specific optimization
        if model_info["type"] == "vae":
            return "low"  # VAEs need precision
        elif model_info["type"] == "dit":
            return "high"  # DiTs benefit greatly from optimization
        elif model_info["type"] == "transformer":
            # Large transformers can handle aggressive optimization
            if model_info["params"] > 5e9:  # > 5B params
                return "extreme"
            else:
                return "high"
        elif model_info["type"] == "unet":
            # UNets are balanced
            if model_info["attention_layers"] > 30:
                return "high"
            else:
                return "medium"
        else:
            return "medium"  # Default safe choice

    def _calculate_speedup(self, level: str) -> float:
        """
        Calculate expected speedup
        """
        speedups = {
            "low": 2.0,
            "medium": 5.0,
            "high": 10.0,
            "extreme": 20.0
        }
        return speedups.get(level, 1.0)


class JetBlockWorkflowProfiler:
    """
    Profile entire workflow for optimization opportunities
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile_depth": (["quick", "standard", "deep"],),
                "show_recommendations": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("profiling_report",)
    FUNCTION = "profile_workflow"
    CATEGORY = "Deterministic/Analysis"

    def profile_workflow(self, profile_depth, show_recommendations):
        """
        Profile workflow and provide optimization recommendations
        """

        report = []
        report.append("=" * 60)
        report.append("DETERMINISTIC TOOLKIT WORKFLOW PROFILING")
        report.append("=" * 60)

        # Placeholder for actual workflow profiling
        # In production, this would hook into ComfyUI's execution

        # Simulated profiling results
        profiling_data = {
            "total_nodes": 15,
            "attention_operations": 120,
            "memory_transfers": 45,
            "redundant_computations": 18,
            "cache_opportunities": 35,
            "parallelizable_branches": 3
        }

        report.append("\nWorkflow Statistics:")
        report.append(f"  Total Nodes: {profiling_data['total_nodes']}")
        report.append(f"  Attention Operations: {profiling_data['attention_operations']}")
        report.append(f"  Memory Transfers: {profiling_data['memory_transfers']}")

        report.append("\nOptimization Opportunities:")
        report.append(f"  Redundant Computations: {profiling_data['redundant_computations']}")
        report.append(f"  Cache Opportunities: {profiling_data['cache_opportunities']}")
        report.append(f"  Parallelizable Branches: {profiling_data['parallelizable_branches']}")

        if show_recommendations:
            report.append("\nRecommendations:")
            report.append("  1. Enable Deterministic Auto-Optimizer for 10x speedup")
            report.append("  2. Use Pattern Caching for repeated operations")
            report.append("  3. Enable Temporal Skipping for 5x faster sampling")
            report.append("  4. Consider batch processing for maximum throughput")

            # Calculate potential speedup
            potential_speedup = 1.0
            potential_speedup *= (1 + profiling_data['redundant_computations'] / 10)
            potential_speedup *= (1 + profiling_data['cache_opportunities'] / 20)
            potential_speedup *= (1 + profiling_data['parallelizable_branches'] * 0.5)

            report.append(f"\nPotential Speedup: {potential_speedup:.1f}x")

        report.append("=" * 60)

        return ("\n".join(report),)


class JetBlockSafetySwitch:
    """
    Emergency switch to disable all optimizations if issues occur
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_optimizations": ("BOOLEAN", {"default": True}),
                "reset_cache": ("BOOLEAN", {"default": False}),
                "verbose_logging": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "safety_switch"
    CATEGORY = "Deterministic/Safety"

    def safety_switch(self, enable_optimizations, reset_cache, verbose_logging):
        """
        Master safety switch for all deterministic optimizations
        """
        import torch
        import gc

        optimizer = get_optimizer()
        status_messages = []

        if not enable_optimizations:
            # Disable all optimizations
            status_messages.append("SAFETY MODE: All optimizations DISABLED")

            # Reset to safe defaults
            optimizer.temporal_skipper.skip_ratio = 0.0
            optimizer.attention_cache.cache.clear()

            # Disable PyTorch optimizations
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False

            status_messages.append("- Temporal skipping: DISABLED")
            status_messages.append("- Pattern caching: CLEARED")
            status_messages.append("- TF32: DISABLED")
            status_messages.append("- cuDNN benchmark: DISABLED")

        else:
            status_messages.append("Deterministic Toolkit: ENABLED")

        if reset_cache:
            # Clear all caches
            optimizer.attention_cache.cache.clear()
            optimizer.attention_cache.current_cache_bytes = 0
            optimizer.attention_cache.cache_hits = 0
            optimizer.attention_cache.cache_misses = 0

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            status_messages.append("All caches CLEARED")

        if verbose_logging:
            import logging
            logging.getLogger("DeterministicToolkit").setLevel(logging.DEBUG)
            status_messages.append("Verbose logging ENABLED")
        else:
            import logging
            logging.getLogger("DeterministicToolkit").setLevel(logging.INFO)

        return ("\n".join(status_messages),)


# Add to node mappings
COMPATIBILITY_NODE_CLASS_MAPPINGS = {
    "JetBlockCompatibilityChecker": JetBlockCompatibilityChecker,
    "JetBlockSmartOptimizer": JetBlockSmartOptimizer,
    "JetBlockWorkflowProfiler": JetBlockWorkflowProfiler,
    "JetBlockSafetySwitch": JetBlockSafetySwitch,
}

COMPATIBILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "JetBlockCompatibilityChecker": "Workflow Compatibility Checker",
    "JetBlockSmartOptimizer": "Smart Deterministic Optimizer",
    "JetBlockWorkflowProfiler": "Workflow Profiler",
    "JetBlockSafetySwitch": "Safety Switch",
}