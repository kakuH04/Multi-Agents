"""Main coordinator for managing tasks across multiple worker Macs"""
from flask import Flask, request, jsonify
from typing import Dict, List, Optional
import uuid
import logging
from datetime import datetime
import json
from pathlib import Path
import requests
from threading import Thread, Lock

from ..common.task import Task, TaskStatus, TaskType
from ..common.config import Config
from ..git_manager.worktree_manager import WorktreeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCoordinator:
    """Coordinates tasks across multiple worker Macs"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.tasks: Dict[str, Task] = {}
        self.workers: Dict[str, Dict] = {}  # worker_id -> worker info
        self.task_lock = Lock()

        # Setup result directory
        self.result_dir = Path(self.config.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Flask app for API
        self.app = Flask(__name__)
        self._setup_routes()

        # Load workers from config
        self._load_workers()

    def _load_workers(self):
        """Load worker configurations"""
        worker_configs = self.config.workers
        for idx, worker_config in enumerate(worker_configs):
            worker_id = worker_config.get('name', f"worker-{idx}")
            self.workers[worker_id] = {
                'id': worker_id,
                'host': worker_config.get('host'),
                'port': worker_config.get('port', 5001),
                'ssh_user': worker_config.get('ssh_user'),
                'ssh_key': worker_config.get('ssh_key'),
                'max_concurrent': worker_config.get('max_concurrent_tasks', 3),
                'status': 'unknown',
                'active_tasks': []
            }
        logger.info(f"Loaded {len(self.workers)} workers")

    def _setup_routes(self):
        """Setup Flask API routes"""

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

        @self.app.route('/tasks', methods=['POST'])
        def create_task():
            """Create a new task"""
            data = request.json
            task = self.create_task(
                task_type=TaskType(data['task_type']),
                name=data['name'],
                description=data.get('description', ''),
                script_path=data['script_path'],
                parameters=data.get('parameters', {})
            )
            return jsonify(task.to_dict()), 201

        @self.app.route('/tasks/<task_id>', methods=['GET'])
        def get_task(task_id):
            """Get task status"""
            task = self.tasks.get(task_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            return jsonify(task.to_dict())

        @self.app.route('/tasks', methods=['GET'])
        def list_tasks():
            """List all tasks"""
            return jsonify([task.to_dict() for task in self.tasks.values()])

        @self.app.route('/tasks/<task_id>/result', methods=['POST'])
        def update_task_result():
            """Receive task result from worker"""
            data = request.json
            task_id = data.get('task_id')
            result = data.get('result')
            metrics = data.get('metrics', {})
            resource_usage = data.get('resource_usage', {})
            status = data.get('status', 'completed')
            error = data.get('error')

            self.update_task_result(task_id, result, metrics, resource_usage, status, error)
            return jsonify({'status': 'updated'})

        @self.app.route('/workers', methods=['GET'])
        def list_workers():
            """List all workers"""
            return jsonify(list(self.workers.values()))

    def create_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        script_path: str,
        parameters: Dict = None
    ) -> Task:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            name=name,
            description=description,
            script_path=script_path,
            parameters=parameters or {}
        )

        with self.task_lock:
            self.tasks[task_id] = task

        logger.info(f"Created task {task_id}: {name}")
        return task

    def assign_task(self, task_id: str) -> Optional[str]:
        """Assign task to an available worker"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return None

        # Find available worker
        worker_id = self._find_available_worker()
        if not worker_id:
            logger.warning("No available workers")
            return None

        # Update task
        task.status = TaskStatus.ASSIGNED
        task.worker_id = worker_id
        task.worktree_name = f"task-{task.name}-{task_id[:8]}"

        # Add to worker's active tasks
        self.workers[worker_id]['active_tasks'].append(task_id)

        # Send task to worker
        self._send_task_to_worker(worker_id, task)

        logger.info(f"Assigned task {task_id} to worker {worker_id}")
        return worker_id

    def _find_available_worker(self) -> Optional[str]:
        """Find an available worker that can take more tasks"""
        for worker_id, worker in self.workers.items():
            if len(worker['active_tasks']) < worker['max_concurrent']:
                return worker_id
        return None

    def _send_task_to_worker(self, worker_id: str, task: Task):
        """Send task to worker via HTTP"""
        worker = self.workers[worker_id]
        url = f"http://{worker['host']}:{worker['port']}/execute"

        try:
            response = requests.post(url, json=task.to_dict(), timeout=5)
            if response.status_code == 200:
                logger.info(f"Task {task.task_id} sent to worker {worker_id}")
            else:
                logger.error(f"Failed to send task to worker: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending task to worker {worker_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)

    def update_task_result(
        self,
        task_id: str,
        result: Dict,
        metrics: Dict[str, float],
        resource_usage: Dict,
        status: str = 'completed',
        error: Optional[str] = None
    ):
        """Update task with results from worker"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return

            task.result = result
            task.metrics = metrics
            task.resource_usage = resource_usage
            task.status = TaskStatus(status)
            task.completed_at = datetime.now()
            task.error = error

            # Remove from worker's active tasks
            if task.worker_id and task.worker_id in self.workers:
                worker = self.workers[task.worker_id]
                if task_id in worker['active_tasks']:
                    worker['active_tasks'].remove(task_id)

            # Save result to file
            self._save_result(task)

            logger.info(f"Updated task {task_id} result: {status}")

    def _save_result(self, task: Task):
        """Save task result to file"""
        result_file = self.result_dir / f"{task.task_id}.json"
        with open(result_file, 'w') as f:
            json.dump(task.to_dict(), indent=2, fp=f)

    def get_completed_tasks(self, task_type: Optional[TaskType] = None) -> List[Task]:
        """Get all completed tasks, optionally filtered by type"""
        completed = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]

        if task_type:
            completed = [t for t in completed if t.task_type == task_type]

        return completed

    def select_best_task(
        self,
        tasks: List[Task],
        metric_key: str = 'accuracy',
        higher_is_better: bool = True
    ) -> Optional[Task]:
        """Select the best task based on a metric"""
        if not tasks:
            return None

        # Filter tasks with the metric
        tasks_with_metric = [t for t in tasks if metric_key in t.metrics]

        if not tasks_with_metric:
            logger.warning(f"No tasks have metric '{metric_key}'")
            return None

        # Sort by metric
        sorted_tasks = sorted(
            tasks_with_metric,
            key=lambda t: t.metrics[metric_key],
            reverse=higher_is_better
        )

        best_task = sorted_tasks[0]
        logger.info(f"Best task: {best_task.task_id} with {metric_key}={best_task.metrics[metric_key]}")

        return best_task

    def cleanup_except_best(
        self,
        tasks: List[Task],
        best_task: Task,
        cleanup_worktrees: bool = True,
        cleanup_models: bool = True
    ):
        """Clean up all resources except the best task"""
        worktree_manager = WorktreeManager(
            base_repo_path=".",
            worktree_base=self.config.worktree_base
        )

        for task in tasks:
            if task.task_id == best_task.task_id:
                continue

            logger.info(f"Cleaning up task {task.task_id}")

            # Remove worktree
            if cleanup_worktrees and task.worktree_name:
                try:
                    worktree_manager.remove_worktree(task.worktree_name, force=True)
                except Exception as e:
                    logger.error(f"Failed to remove worktree: {e}")

            # Remove model files if specified in result
            if cleanup_models and task.result and 'model_path' in task.result:
                model_path = Path(task.result['model_path'])
                if model_path.exists():
                    try:
                        if model_path.is_file():
                            model_path.unlink()
                        elif model_path.is_dir():
                            import shutil
                            shutil.rmtree(model_path)
                        logger.info(f"Removed model at {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove model: {e}")

    def run(self):
        """Start the coordinator server"""
        logger.info(f"Starting coordinator on {self.config.management_host}:{self.config.management_port}")
        self.app.run(
            host=self.config.management_host,
            port=self.config.management_port,
            debug=False
        )
