"""
Deterministic Toolkit Compatibility Checker and Auto-Adjuster
Automatically detects workflow types and adjusts optimization levels
"""

import torch
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("DeterministicToolkit.Compatibility")

@dataclass
class CompatibilityProfile:
    """Profile for different workflow types"""
    name: str
    optimization_level: str  # low, medium, high, extreme
    use_temporal_skip: bool
    use_pattern_cache: bool
    skip_ratio: float
    safe_nodes: List[str]  # Nodes that are safe to optimize
    unsafe_nodes: List[str]  # Nodes that should not be optimized
    warnings: List[str]


class WorkflowAnalyzer:
    """Analyzes workflows for compatibility issues"""

    def __init__(self):
        self.profiles = self._initialize_profiles()
        self.node_compatibility = self._initialize_node_compatibility()
        self.detection_patterns = self._initialize_detection_patterns()

    def _initialize_profiles(self) -> Dict[str, CompatibilityProfile]:
        """Initialize compatibility profiles for different workflow types"""
        return {
            "standard": CompatibilityProfile(
                name="Standard Generation",
                optimization_level="high",
                use_temporal_skip=True,
                use_pattern_cache=True,
                skip_ratio=0.8,
                safe_nodes=["all"],
                unsafe_nodes=[],
                warnings=[]
            ),
            "animatediff": CompatibilityProfile(
                name="AnimateDiff",
                optimization_level="low",
                use_temporal_skip=False,  # AnimateDiff has its own temporal handling
                use_pattern_cache=True,
                skip_ratio=0.0,
                safe_nodes=["CLIPTextEncode", "VAEDecode", "VAEEncode"],
                unsafe_nodes=["AnimateDiffEvolvedSampling", "AnimateDiffLoaderWithContext"],
                warnings=["AnimateDiff detected: Temporal skipping disabled for consistency"]
            ),
            "controlnet": CompatibilityProfile(
                name="ControlNet",
                optimization_level="medium",
                use_temporal_skip=True,
                use_pattern_cache=True,
                skip_ratio=0.5,  # Lower skip ratio for precision
                safe_nodes=["all"],
                unsafe_nodes=["ControlNetApplyAdvanced"],
                warnings=["ControlNet detected: Using medium optimization for precision"]
            ),
            "video": CompatibilityProfile(
                name="Video Generation",
                optimization_level="medium",
                use_temporal_skip=False,
                use_pattern_cache=True,
                skip_ratio=0.3,
                safe_nodes=["CLIPTextEncode", "CheckpointLoaderSimple"],
                unsafe_nodes=["VideoCombine", "VideoLinearCFGGuidance"],
                warnings=["Video workflow: Temporal coherence limited to preserve frame consistency"]
            ),
            "upscale": CompatibilityProfile(
                name="Upscaling",
                optimization_level="low",
                use_temporal_skip=False,
                use_pattern_cache=False,  # Upscalers need precise patterns
                skip_ratio=0.0,
                safe_nodes=["LoadImage", "SaveImage"],
                unsafe_nodes=["UpscaleModelLoader", "ImageUpscaleWithModel"],
                warnings=["Upscaling workflow: Minimal optimization to preserve detail"]
            ),
            "complex": CompatibilityProfile(
                name="Complex Multi-Stage",
                optimization_level="medium",
                use_temporal_skip=True,
                use_pattern_cache=True,
                skip_ratio=0.6,
                safe_nodes=["all"],
                unsafe_nodes=[],
                warnings=["Complex workflow detected: Using balanced optimization"]
            ),
            "ipadapter": CompatibilityProfile(
                name="IPAdapter",
                optimization_level="high",
                use_temporal_skip=True,
                use_pattern_cache=True,
                skip_ratio=0.7,
                safe_nodes=["all"],
                unsafe_nodes=["IPAdapterApplyFaceID"],
                warnings=[]
            ),
            "tensorrt": CompatibilityProfile(
                name="TensorRT",
                optimization_level="low",  # TensorRT already optimized
                use_temporal_skip=False,
                use_pattern_cache=False,
                skip_ratio=0.0,
                safe_nodes=["CLIPTextEncode"],
                unsafe_nodes=["TensorRTLoader", "VAEDecode_TensorRT"],
                warnings=["TensorRT detected: deterministic optimization reduced to avoid conflicts"]
            ),
            "custom": CompatibilityProfile(
                name="Custom Nodes Heavy",
                optimization_level="low",
                use_temporal_skip=False,
                use_pattern_cache=True,
                skip_ratio=0.3,
                safe_nodes=["KSampler", "CLIPTextEncode", "VAEDecode"],
                unsafe_nodes=[],
                warnings=["Many custom nodes detected: Using conservative optimization"]
            ),
            "experimental": CompatibilityProfile(
                name="Experimental",
                optimization_level="low",
                use_temporal_skip=False,
                use_pattern_cache=False,
                skip_ratio=0.0,
                safe_nodes=["LoadImage", "SaveImage"],
                unsafe_nodes=["all"],
                warnings=["Experimental nodes detected: Minimal optimization for stability"]
            )
        }

    def _initialize_node_compatibility(self) -> Dict[str, str]:
        """Map node types to compatibility levels"""
        return {
            # Fully compatible (can use all optimizations)
            "KSampler": "full",
            "KSamplerAdvanced": "full",
            "CLIPTextEncode": "full",
            "VAEDecode": "full",
            "VAEEncode": "full",
            "CheckpointLoaderSimple": "full",
            "LoraLoader": "full",
            "EmptyLatentImage": "full",
            "LatentUpscale": "full",

            # Partial compatibility (some optimizations may cause issues)
            "ControlNetApply": "partial",
            "ControlNetApplyAdvanced": "partial",
            "AnimateDiffLoaderWithContext": "partial",
            "AnimateDiffEvolvedSampling": "partial",
            "IPAdapterApply": "partial",
            "FaceDetailer": "partial",

            # Minimal compatibility (use conservative settings)
            "VideoLinearCFGGuidance": "minimal",
            "VideoCombine": "minimal",
            "UpscaleModelLoader": "minimal",
            "ImageUpscaleWithModel": "minimal",
            "UltimateSDUpscale": "minimal",

            # Incompatible (disable optimizations)
            "TensorRTLoader": "incompatible",
            "VAEDecode_TensorRT": "incompatible",
            "KSamplerAdvanced_TensorRT": "incompatible",
        }

    def _initialize_detection_patterns(self) -> Dict[str, List[str]]:
        """Patterns to detect workflow types"""
        return {
            "animatediff": ["AnimateDiff", "MotionModel", "AnimateDiffEvolve"],
            "controlnet": ["ControlNet", "ControlNetApply", "ControlNetLoader"],
            "video": ["Video", "Frame", "VideoLinear", "VideoCombine"],
            "upscale": ["Upscale", "ESRGAN", "UltimateSD", "HighRes"],
            "tensorrt": ["TensorRT", "_TensorRT", "TRT"],
            "ipadapter": ["IPAdapter", "FaceID", "InstantID"],
            "custom": ["ComfyUI-", "WAS_", "CR_", "Efficiency", "Impact"],
            "experimental": ["Test", "Beta", "Experimental", "Debug"]
        }

    def analyze_workflow(self, workflow: Dict[str, Any]) -> CompatibilityProfile:
        """Analyze workflow and return appropriate compatibility profile"""

        # Extract node types from workflow
        node_types = []
        if "nodes" in workflow:
            for node in workflow["nodes"]:
                if "type" in node:
                    node_types.append(node["type"])
                elif "class_type" in node:
                    node_types.append(node["class_type"])

        # Detect workflow type
        detected_types = []
        for workflow_type, patterns in self.detection_patterns.items():
            for pattern in patterns:
                if any(pattern in node_type for node_type in node_types):
                    detected_types.append(workflow_type)
                    break

        # Count incompatible nodes
        incompatible_count = 0
        partial_count = 0
        for node_type in node_types:
            compatibility = self.node_compatibility.get(node_type, "unknown")
            if compatibility == "incompatible":
                incompatible_count += 1
            elif compatibility == "partial":
                partial_count += 1

        # Determine profile based on detection
        if "tensorrt" in detected_types:
            return self.profiles["tensorrt"]
        elif "animatediff" in detected_types:
            return self.profiles["animatediff"]
        elif "video" in detected_types:
            return self.profiles["video"]
        elif "upscale" in detected_types:
            return self.profiles["upscale"]
        elif "controlnet" in detected_types:
            return self.profiles["controlnet"]
        elif "ipadapter" in detected_types:
            return self.profiles["ipadapter"]
        elif incompatible_count > 2:
            return self.profiles["experimental"]
        elif len(detected_types) > 3 or partial_count > 5:
            return self.profiles["complex"]
        elif any("ComfyUI-" in node for node in node_types):
            return self.profiles["custom"]
        else:
            return self.profiles["standard"]

    def get_node_optimization_level(self, node_type: str, profile: CompatibilityProfile) -> str:
        """Get optimization level for specific node"""

        # Check if node is in unsafe list
        if node_type in profile.unsafe_nodes:
            return "disabled"

        # Check if all nodes are safe
        if "all" in profile.safe_nodes:
            return profile.optimization_level

        # Check if node is explicitly safe
        if node_type in profile.safe_nodes:
            return profile.optimization_level

        # Check compatibility level
        compatibility = self.node_compatibility.get(node_type, "unknown")

        if compatibility == "incompatible":
            return "disabled"
        elif compatibility == "minimal":
            return "low"
        elif compatibility == "partial":
            return "medium" if profile.optimization_level in ["high", "extreme"] else profile.optimization_level
        else:
            return profile.optimization_level


class CompatibilityReport:
    """Generate detailed compatibility report"""

    def __init__(self, profile: CompatibilityProfile, workflow_analysis: Dict[str, Any]):
        self.profile = profile
        self.analysis = workflow_analysis

    def generate_report(self) -> str:
        """Generate human-readable compatibility report"""

        report = []
        report.append("=" * 60)
        report.append("DETERMINISTIC TOOLKIT COMPATIBILITY REPORT")
        report.append("=" * 60)
        report.append(f"Workflow Type: {self.profile.name}")
        report.append(f"Optimization Level: {self.profile.optimization_level.upper()}")
        report.append(f"Temporal Skip: {'ENABLED' if self.profile.use_temporal_skip else 'DISABLED'}")
        report.append(f"Pattern Cache: {'ENABLED' if self.profile.use_pattern_cache else 'DISABLED'}")

        if self.profile.use_temporal_skip:
            report.append(f"Skip Ratio: {self.profile.skip_ratio:.0%}")

        if self.profile.warnings:
            report.append("\nWarnings:")
            for warning in self.profile.warnings:
                report.append(f"  ! {warning}")

        if self.analysis:
            report.append("\nWorkflow Analysis:")
            report.append(f"  Total Nodes: {self.analysis.get('total_nodes', 0)}")
            report.append(f"  Optimizable: {self.analysis.get('optimizable_nodes', 0)}")
            report.append(f"  Incompatible: {self.analysis.get('incompatible_nodes', 0)}")

            expected_speedup = self._calculate_expected_speedup()
            report.append(f"\nExpected Speedup: {expected_speedup:.1f}x")

        report.append("=" * 60)

        return "\n".join(report)

    def _calculate_expected_speedup(self) -> float:
        """Calculate expected speedup based on profile"""

        base_speedup = {
            "disabled": 1.0,
            "low": 2.0,
            "medium": 5.0,
            "high": 10.0,
            "extreme": 20.0
        }

        speedup = base_speedup.get(self.profile.optimization_level, 1.0)

        if self.profile.use_temporal_skip:
            speedup *= (1 + self.profile.skip_ratio)

        if self.profile.use_pattern_cache:
            speedup *= 1.5

        return speedup


# Global analyzer instance
WORKFLOW_ANALYZER = WorkflowAnalyzer()

def check_compatibility(workflow: Dict[str, Any]) -> Tuple[CompatibilityProfile, str]:
    """Main function to check workflow compatibility"""

    profile = WORKFLOW_ANALYZER.analyze_workflow(workflow)

    # Generate analysis
    analysis = {
        "total_nodes": len(workflow.get("nodes", [])),
        "optimizable_nodes": 0,
        "incompatible_nodes": 0
    }

    for node in workflow.get("nodes", []):
        node_type = node.get("type") or node.get("class_type", "")
        opt_level = WORKFLOW_ANALYZER.get_node_optimization_level(node_type, profile)

        if opt_level == "disabled":
            analysis["incompatible_nodes"] += 1
        elif opt_level != "disabled":
            analysis["optimizable_nodes"] += 1

    # Generate report
    report_gen = CompatibilityReport(profile, analysis)
    report = report_gen.generate_report()

    return profile, report