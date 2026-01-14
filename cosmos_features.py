"""
Cosmos-Inspired Features for JetBlock Optimizer
Predictive loading, pattern learning, and intelligent optimization
"""

import torch
import numpy as np
import time
import json
import os
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle
import logging

logger = logging.getLogger("JetBlock.Cosmos")

# Try to import NVIDIA ML for temperature monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    logger.info("NVIDIA ML monitoring enabled")
except:
    NVML_AVAILABLE = False
    logger.info("NVIDIA ML not available - temperature monitoring disabled")


@dataclass
class WorkflowPattern:
    """Pattern learned from workflow execution"""
    sequence: List[str]
    frequency: int = 1
    avg_execution_time: float = 0.0
    model_sequence: List[str] = field(default_factory=list)
    next_likely_nodes: Dict[str, float] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)


class CosmosPredictor:
    """
    Cosmos-inspired predictive engine
    Learns from your workflows and predicts next operations
    """

    def __init__(self, history_size: int = 1000):
        self.history = deque(maxlen=history_size)
        self.patterns = {}
        self.model_load_times = {}
        self.node_transitions = defaultdict(lambda: defaultdict(int))
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cosmos_cache")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load existing patterns
        self.load_patterns()

    def record_node_execution(self, node_type: str, execution_time: float,
                            model_used: Optional[str] = None):
        """Record node execution for pattern learning"""

        self.history.append({
            "node": node_type,
            "time": execution_time,
            "model": model_used,
            "timestamp": time.time()
        })

        # Update transitions
        if len(self.history) > 1:
            prev_node = self.history[-2]["node"]
            self.node_transitions[prev_node][node_type] += 1

        # Learn patterns every 10 executions
        if len(self.history) % 10 == 0:
            self.learn_patterns()

    def learn_patterns(self):
        """Learn patterns from execution history"""

        # Extract sequences of length 3-5
        for seq_len in range(3, 6):
            for i in range(len(self.history) - seq_len):
                sequence = [self.history[j]["node"] for j in range(i, i + seq_len)]
                seq_key = tuple(sequence)

                if seq_key not in self.patterns:
                    self.patterns[seq_key] = WorkflowPattern(sequence=sequence)
                else:
                    self.patterns[seq_key].frequency += 1

                # Update average execution time
                total_time = sum(self.history[j]["time"] for j in range(i, i + seq_len))
                self.patterns[seq_key].avg_execution_time = (
                    (self.patterns[seq_key].avg_execution_time *
                     (self.patterns[seq_key].frequency - 1) + total_time) /
                    self.patterns[seq_key].frequency
                )

        # Save patterns periodically
        if len(self.patterns) % 50 == 0:
            self.save_patterns()

    def predict_next_nodes(self, current_sequence: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict most likely next nodes based on patterns"""

        predictions = defaultdict(float)

        # Check all patterns that start with current sequence
        for pattern_key, pattern in self.patterns.items():
            pattern_list = list(pattern_key)

            # Check if current sequence matches beginning of pattern
            if len(current_sequence) <= len(pattern_list):
                match = all(
                    current_sequence[i] == pattern_list[i]
                    for i in range(len(current_sequence))
                )

                if match and len(current_sequence) < len(pattern_list):
                    next_node = pattern_list[len(current_sequence)]
                    # Weight by pattern frequency
                    predictions[next_node] += pattern.frequency

        # Also use transition probabilities
        if current_sequence:
            last_node = current_sequence[-1]
            for next_node, count in self.node_transitions[last_node].items():
                predictions[next_node] += count * 0.5  # Lower weight for simple transitions

        # Normalize and sort
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v/total for k, v in predictions.items()}

        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def predict_models_needed(self, predicted_nodes: List[str]) -> List[str]:
        """Predict which models will be needed based on predicted nodes"""

        model_predictions = set()

        node_model_map = {
            "KSampler": "checkpoint",
            "KSamplerAdvanced": "checkpoint",
            "VAEDecode": "vae",
            "VAEEncode": "vae",
            "ControlNetApply": "controlnet",
            "LoraLoader": "lora",
            "CLIPTextEncode": "clip",
        }

        for node, _ in predicted_nodes:
            if node in node_model_map:
                model_predictions.add(node_model_map[node])

        return list(model_predictions)

    def save_patterns(self):
        """Save learned patterns to disk"""

        pattern_file = os.path.join(self.cache_dir, "workflow_patterns.pkl")

        try:
            with open(pattern_file, 'wb') as f:
                pickle.dump({
                    'patterns': self.patterns,
                    'transitions': dict(self.node_transitions),
                    'history': list(self.history)
                }, f)
            logger.debug(f"Saved {len(self.patterns)} patterns")
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

    def load_patterns(self):
        """Load patterns from disk"""

        pattern_file = os.path.join(self.cache_dir, "workflow_patterns.pkl")

        if os.path.exists(pattern_file):
            try:
                with open(pattern_file, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data.get('patterns', {})
                    self.node_transitions = defaultdict(
                        lambda: defaultdict(int),
                        data.get('transitions', {})
                    )
                    self.history = deque(data.get('history', []), maxlen=1000)
                logger.info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")


class CosmosInterpolator:
    """
    Frame interpolation for multi-step workflows
    Skip intermediate steps by interpolating
    """

    def __init__(self):
        self.interpolation_cache = {}
        self.quality_threshold = 0.95

    def can_interpolate(self, start_latent: torch.Tensor, end_latent: torch.Tensor,
                       num_steps: int) -> bool:
        """Check if interpolation is safe for these latents"""

        # Check dimensionality
        if start_latent.shape != end_latent.shape:
            return False

        # Check if latents are too different (would cause artifacts)
        diff = torch.norm(end_latent - start_latent)
        max_safe_diff = 10.0 * num_steps  # Heuristic threshold

        return diff < max_safe_diff

    def interpolate_latents(self, start: torch.Tensor, end: torch.Tensor,
                          num_intermediate: int) -> List[torch.Tensor]:
        """Interpolate between latents for skipped steps"""

        intermediates = []

        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)

            # Spherical linear interpolation for better quality
            interpolated = self.slerp(start, end, alpha)
            intermediates.append(interpolated)

        return intermediates

    def slerp(self, start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical linear interpolation"""

        # Flatten tensors
        start_flat = start.view(start.size(0), -1)
        end_flat = end.view(end.size(0), -1)

        # Normalize
        start_norm = start_flat / start_flat.norm(dim=1, keepdim=True)
        end_norm = end_flat / end_flat.norm(dim=1, keepdim=True)

        # Compute angle
        dot = (start_norm * end_norm).sum(dim=1, keepdim=True)
        dot = torch.clamp(dot, -1, 1)
        theta = torch.acos(dot)

        # Interpolate
        sin_theta = torch.sin(theta)

        # Handle edge case where vectors are parallel
        mask = sin_theta.abs() < 1e-6

        result = torch.where(
            mask,
            (1 - alpha) * start_flat + alpha * end_flat,  # Linear interpolation
            (torch.sin((1 - alpha) * theta) / sin_theta) * start_flat +
            (torch.sin(alpha * theta) / sin_theta) * end_flat  # Spherical interpolation
        )

        return result.view_as(start)


class IntelligentModelPreloader:
    """
    Preload models based on predictions
    Eliminates loading delays
    """

    def __init__(self, max_preload_gb: float = 8.0):
        self.max_preload_bytes = max_preload_gb * 1024 * 1024 * 1024
        self.preloaded_models = {}
        self.model_sizes = {}
        self.load_times = {}
        self.predictor = CosmosPredictor()

    def predict_and_preload(self, current_node: str):
        """Predict and preload models for upcoming nodes"""

        # Get current sequence
        recent_nodes = [h["node"] for h in list(self.predictor.history)[-5:]]
        recent_nodes.append(current_node)

        # Predict next nodes
        predicted_nodes = self.predictor.predict_next_nodes(recent_nodes)

        # Predict models needed
        models_needed = self.predictor.predict_models_needed(predicted_nodes)

        # Preload models
        for model_type in models_needed:
            if model_type not in self.preloaded_models:
                self.preload_model_async(model_type)

    def preload_model_async(self, model_type: str):
        """Asynchronously preload a model"""

        import threading

        def load_worker():
            try:
                logger.info(f"Preloading {model_type}...")
                # Simulate model loading (in reality, would load actual model)
                time.sleep(0.1)  # Placeholder
                self.preloaded_models[model_type] = f"Preloaded_{model_type}"
                logger.info(f"Preloaded {model_type} successfully")
            except Exception as e:
                logger.error(f"Failed to preload {model_type}: {e}")

        thread = threading.Thread(target=load_worker, daemon=True)
        thread.start()

    def get_model(self, model_type: str):
        """Get preloaded model or load on demand"""

        if model_type in self.preloaded_models:
            logger.info(f"Using preloaded {model_type}")
            return self.preloaded_models[model_type]
        else:
            logger.info(f"Loading {model_type} on demand")
            # Load model normally
            return None

    def clear_cache(self):
        """Clear preloaded models"""

        self.preloaded_models.clear()
        torch.cuda.empty_cache()


class GPUMonitor:
    """
    Monitor GPU temperature and performance
    """

    def __init__(self):
        self.temps = deque(maxlen=100)
        self.power = deque(maxlen=100)
        self.memory = deque(maxlen=100)
        self.utilization = deque(maxlen=100)

        if NVML_AVAILABLE:
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # RTX 4090
            logger.info(f"GPU monitoring active for {pynvml.nvmlDeviceGetName(self.handle)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""

        stats = {
            "available": NVML_AVAILABLE,
            "temperature": 0,
            "power": 0,
            "memory_used": 0,
            "memory_total": 0,
            "utilization": 0,
            "fan_speed": 0
        }

        if NVML_AVAILABLE:
            try:
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                stats["temperature"] = temp
                self.temps.append(temp)

                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to W
                stats["power"] = power
                self.power.append(power)

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                stats["memory_used"] = mem_info.used / (1024**3)  # GB
                stats["memory_total"] = mem_info.total / (1024**3)  # GB
                self.memory.append(stats["memory_used"])

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                stats["utilization"] = util.gpu
                self.utilization.append(util.gpu)

                # Fan speed
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(self.handle)
                    stats["fan_speed"] = fan
                except:
                    pass

            except Exception as e:
                logger.error(f"Error getting GPU stats: {e}")

        return stats

    def get_health_status(self) -> Tuple[str, str]:
        """Get GPU health status"""

        stats = self.get_stats()

        if not stats["available"]:
            return "UNKNOWN", "GPU monitoring not available"

        temp = stats["temperature"]
        power = stats["power"]

        # Temperature thresholds for RTX 4090
        if temp < 70:
            temp_status = "EXCELLENT"
        elif temp < 80:
            temp_status = "GOOD"
        elif temp < 85:
            temp_status = "WARNING"
        else:
            temp_status = "CRITICAL"

        # Power status (RTX 4090 TDP is 450W)
        if power < 300:
            power_status = "Efficient"
        elif power < 400:
            power_status = "Normal"
        else:
            power_status = "High"

        message = f"Temp: {temp}°C ({temp_status}), Power: {power:.0f}W ({power_status})"

        return temp_status, message

    def get_performance_score(self) -> float:
        """Calculate performance score (0-100)"""

        if not self.temps:
            return 100.0

        # Calculate based on temperature (lower is better)
        avg_temp = sum(self.temps) / len(self.temps)
        temp_score = max(0, 100 - (avg_temp - 50) * 2)

        # Calculate based on utilization (higher is better)
        avg_util = sum(self.utilization) / len(self.utilization) if self.utilization else 0
        util_score = avg_util

        # Combined score
        return (temp_score * 0.3 + util_score * 0.7)


class WorkflowOptimizationEngine:
    """
    Self-optimizing workflow engine
    Learns and improves over time
    """

    def __init__(self):
        self.predictor = CosmosPredictor()
        self.interpolator = CosmosInterpolator()
        self.preloader = IntelligentModelPreloader()
        self.gpu_monitor = GPUMonitor()
        self.optimization_history = []

    def analyze_and_optimize(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow and return optimization strategy"""

        # Get GPU status
        gpu_stats = self.gpu_monitor.get_stats()
        health_status, health_msg = self.gpu_monitor.get_health_status()

        # Start with base optimization
        optimization = {
            "skip_ratio": 0.5,
            "use_interpolation": True,
            "preload_models": True,
            "cache_size_gb": 8.0,
            "batch_size": 1
        }

        # Adjust based on GPU temperature
        if gpu_stats["temperature"] > 80:
            # Too hot, reduce load
            optimization["skip_ratio"] = 0.3
            optimization["batch_size"] = 1
            logger.warning(f"GPU running hot ({gpu_stats['temperature']}°C), reducing optimization")

        elif gpu_stats["temperature"] < 60:
            # Cool, can be more aggressive
            optimization["skip_ratio"] = 0.8
            optimization["batch_size"] = 4
            logger.info(f"GPU running cool ({gpu_stats['temperature']}°C), maximizing optimization")

        # Adjust based on memory
        mem_free = gpu_stats["memory_total"] - gpu_stats["memory_used"]
        if mem_free > 16:  # >16GB free
            optimization["cache_size_gb"] = 12.0
            optimization["preload_models"] = True
        elif mem_free < 4:  # <4GB free
            optimization["cache_size_gb"] = 2.0
            optimization["preload_models"] = False

        # Learn from patterns
        recent_nodes = [n.get("type", "") for n in workflow.get("nodes", [])][:5]
        predicted_nodes = self.predictor.predict_next_nodes(recent_nodes)

        # Preload predicted models
        if optimization["preload_models"]:
            models_to_preload = self.predictor.predict_models_needed(predicted_nodes)
            for model in models_to_preload:
                self.preloader.preload_model_async(model)

        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "gpu_temp": gpu_stats["temperature"],
            "optimization": optimization,
            "predicted_nodes": predicted_nodes
        })

        return optimization

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics"""

        gpu_stats = self.gpu_monitor.get_stats()
        health_status, health_msg = self.gpu_monitor.get_health_status()
        performance_score = self.gpu_monitor.get_performance_score()

        return {
            "gpu": gpu_stats,
            "health": {
                "status": health_status,
                "message": health_msg,
                "score": performance_score
            },
            "predictions": {
                "patterns_learned": len(self.predictor.patterns),
                "history_size": len(self.predictor.history),
                "preloaded_models": len(self.preloader.preloaded_models)
            },
            "optimization": {
                "total_optimizations": len(self.optimization_history),
                "avg_temperature": sum(self.gpu_monitor.temps) / len(self.gpu_monitor.temps) if self.gpu_monitor.temps else 0,
                "avg_power": sum(self.gpu_monitor.power) / len(self.gpu_monitor.power) if self.gpu_monitor.power else 0
            }
        }


# Global instances
COSMOS_ENGINE = WorkflowOptimizationEngine()

def get_cosmos_engine() -> WorkflowOptimizationEngine:
    """Get global Cosmos engine instance"""
    return COSMOS_ENGINE