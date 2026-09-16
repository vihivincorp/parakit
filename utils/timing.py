"""
Timing and latency measurement utilities for profiling real-time multimodal processing.
"""

import time
import functools
from collections import deque
from typing import Dict, Any, Callable
from .logger import logger

class LatencyTracker:
    """Tracks latency statistics across pipeline stages."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._latencies: Dict[str, deque] = {}
        self._counts: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}

    def record(self, stage_name: str, duration_ms: float):
        if stage_name not in self._latencies:
            self._latencies[stage_name] = deque(maxlen=self.window_size)
            self._counts[stage_name] = 0
        self._latencies[stage_name].append(duration_ms)
        self._counts[stage_name] += 1

    def start(self, key: str):
        self._start_times[key] = time.perf_counter()

    def stop(self, key: str, stage_name: str = None) -> float:
        if key not in self._start_times:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_times.pop(key)) * 1000.0
        target_stage = stage_name if stage_name else key
        self.record(target_stage, elapsed_ms)
        return elapsed_ms

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for stage, durations in self._latencies.items():
            if durations:
                stats[stage] = {
                    "last_ms": round(durations[-1], 2),
                    "avg_ms": round(sum(durations) / len(durations), 2),
                    "min_ms": round(min(durations), 2),
                    "max_ms": round(max(durations), 2),
                    "total_count": self._counts[stage]
                }
        return stats

    def summary_string(self) -> str:
        stats = self.get_stats()
        lines = ["--- Latency Profile (ms) ---"]
        for stage, data in stats.items():
            lines.append(f"  {stage:20s}: avg={data['avg_ms']:6.1f}ms | last={data['last_ms']:6.1f}ms | calls={data['total_count']}")
        return "\n".join(lines)


# Global latency tracker instance
latency_tracker = LatencyTracker()


def time_it(stage_name: str):
    """Decorator to measure function latency and register it with the tracker."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latency_tracker.record(stage_name, elapsed_ms)
        return wrapper
    return decorator
