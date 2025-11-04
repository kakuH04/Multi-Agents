"""Resource monitoring for CPU, GPU, Memory, and Disk"""
import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during task execution"""

    def __init__(self, interval: int = 5, track_gpu: bool = True):
        self.interval = interval
        self.track_gpu = track_gpu and GPU_AVAILABLE
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Resource history
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.disk_usage: List[float] = []
        self.gpu_usage: List[Dict[str, float]] = []
        self.timestamps: List[datetime] = []

    def start(self):
        """Start monitoring resources"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop(self):
        """Stop monitoring resources"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.interval + 1)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._collect_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

    def _collect_metrics(self):
        """Collect current resource metrics"""
        timestamp = datetime.now()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # GPU usage
        gpu_info = []
        if self.track_gpu:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,  # Convert to percentage
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                        'temperature': gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"Could not collect GPU metrics: {e}")

        # Store metrics
        self.timestamps.append(timestamp)
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)
        self.disk_usage.append(disk_percent)
        self.gpu_usage.append(gpu_info)

    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics of resource usage"""
        if not self.cpu_usage:
            return {}

        summary = {
            'cpu': {
                'avg': sum(self.cpu_usage) / len(self.cpu_usage),
                'max': max(self.cpu_usage),
                'min': min(self.cpu_usage)
            },
            'memory': {
                'avg': sum(self.memory_usage) / len(self.memory_usage),
                'max': max(self.memory_usage),
                'min': min(self.memory_usage)
            },
            'disk': {
                'avg': sum(self.disk_usage) / len(self.disk_usage),
                'max': max(self.disk_usage),
                'min': min(self.disk_usage)
            },
            'duration': (self.timestamps[-1] - self.timestamps[0]).total_seconds() if len(self.timestamps) > 1 else 0
        }

        # GPU summary
        if self.track_gpu and self.gpu_usage:
            gpu_summary = {}
            for snapshot in self.gpu_usage:
                for gpu in snapshot:
                    gpu_id = gpu['id']
                    if gpu_id not in gpu_summary:
                        gpu_summary[gpu_id] = {
                            'name': gpu['name'],
                            'load': [],
                            'memory_percent': [],
                            'temperature': []
                        }
                    gpu_summary[gpu_id]['load'].append(gpu['load'])
                    gpu_summary[gpu_id]['memory_percent'].append(gpu['memory_percent'])
                    gpu_summary[gpu_id]['temperature'].append(gpu['temperature'])

            # Calculate averages
            for gpu_id, metrics in gpu_summary.items():
                gpu_summary[gpu_id] = {
                    'name': metrics['name'],
                    'load_avg': sum(metrics['load']) / len(metrics['load']) if metrics['load'] else 0,
                    'load_max': max(metrics['load']) if metrics['load'] else 0,
                    'memory_avg': sum(metrics['memory_percent']) / len(metrics['memory_percent']) if metrics['memory_percent'] else 0,
                    'memory_max': max(metrics['memory_percent']) if metrics['memory_percent'] else 0,
                    'temp_avg': sum(metrics['temperature']) / len(metrics['temperature']) if metrics['temperature'] else 0,
                    'temp_max': max(metrics['temperature']) if metrics['temperature'] else 0
                }

            summary['gpu'] = gpu_summary

        return summary

    def reset(self):
        """Reset collected metrics"""
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.disk_usage.clear()
        self.gpu_usage.clear()
        self.timestamps.clear()

    def get_current_snapshot(self) -> Dict[str, any]:
        """Get current resource usage snapshot"""
        self._collect_metrics()

        snapshot = {
            'timestamp': self.timestamps[-1].isoformat() if self.timestamps else None,
            'cpu_percent': self.cpu_usage[-1] if self.cpu_usage else 0,
            'memory_percent': self.memory_usage[-1] if self.memory_usage else 0,
            'disk_percent': self.disk_usage[-1] if self.disk_usage else 0
        }

        if self.track_gpu and self.gpu_usage:
            snapshot['gpu'] = self.gpu_usage[-1]

        return snapshot
