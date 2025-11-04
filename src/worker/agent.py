"""Worker agent that executes tasks using tmux and monitors resources"""
from flask import Flask, request, jsonify
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict
import json
import time
from threading import Thread
import requests

from ..common.task import Task, TaskStatus
from ..common.config import Config
from ..monitoring.resource_monitor import ResourceMonitor
from ..git_manager.worktree_manager import WorktreeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkerAgent:
    """Worker agent that executes tasks on a Mac"""

    def __init__(self, worker_id: str, config_path: str = "config.yaml"):
        self.worker_id = worker_id
        self.config = Config(config_path)
        self.current_tasks: Dict[str, Task] = {}

        # Initialize worktree manager
        self.worktree_manager = WorktreeManager(
            base_repo_path=".",
            worktree_base=self.config.worktree_base
        )

        # Initialize Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask API routes"""

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'worker_id': self.worker_id,
                'active_tasks': len(self.current_tasks)
            })

        @self.app.route('/execute', methods=['POST'])
        def execute_task():
            """Receive and execute a task"""
            data = request.json
            task = Task.from_dict(data)

            # Execute task in background thread
            thread = Thread(target=self._execute_task, args=(task,))
            thread.daemon = True
            thread.start()

            return jsonify({'status': 'accepted', 'task_id': task.task_id})

        @self.app.route('/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """Get status of a task"""
            task = self.current_tasks.get(task_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            return jsonify(task.to_dict())

    def _execute_task(self, task: Task):
        """Execute a task in a tmux session with resource monitoring"""
        logger.info(f"Executing task {task.task_id}: {task.name}")

        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = None
            self.current_tasks[task.task_id] = task

            # Create worktree
            worktree_path = self.worktree_manager.create_worktree(
                name=task.worktree_name,
                branch=None  # Creates new branch automatically
            )
            logger.info(f"Created worktree at {worktree_path}")

            # Create tmux session
            session_name = f"{self.config.tmux_session_prefix}-{task.task_id[:8]}"
            task.tmux_session = session_name

            self._create_tmux_session(session_name)
            logger.info(f"Created tmux session: {session_name}")

            # Start resource monitoring
            monitor = ResourceMonitor(
                interval=self.config.monitoring_interval,
                track_gpu=self.config.get('monitoring.track_gpu', True)
            )
            monitor.start()

            # Execute the task script in tmux
            script_path = worktree_path / task.script_path
            self._run_in_tmux(
                session_name,
                f"cd {worktree_path} && python {script_path} {self._format_parameters(task.parameters)}"
            )

            # Wait for task completion (monitor output)
            result = self._wait_for_completion(session_name, task)

            # Stop monitoring
            monitor.stop()

            # Get resource summary
            resource_summary = monitor.get_summary()

            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.resource_usage = resource_summary

            # Extract metrics from result if available
            if result and 'metrics' in result:
                task.metrics = result['metrics']

            # Commit changes to worktree
            commit_msg = f"Task {task.name} - {task.task_id}"
            self.worktree_manager.commit_and_push(
                task.worktree_name,
                commit_msg,
                push=False  # Don't push yet, let management decide
            )

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)

        finally:
            # Send results back to management
            self._send_results_to_management(task)

            # Clean up tmux session
            if task.tmux_session:
                self._kill_tmux_session(task.tmux_session)

            # Remove from current tasks
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]

    def _create_tmux_session(self, session_name: str):
        """Create a new tmux session"""
        try:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create tmux session: {e.stderr.decode()}")
            raise

    def _run_in_tmux(self, session_name: str, command: str):
        """Run a command in a tmux session"""
        try:
            subprocess.run(
                ["tmux", "send-keys", "-t", session_name, command, "Enter"],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run command in tmux: {e.stderr.decode()}")
            raise

    def _kill_tmux_session(self, session_name: str):
        """Kill a tmux session"""
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass  # Session may already be dead

    def _wait_for_completion(self, session_name: str, task: Task, timeout: int = 3600) -> Optional[Dict]:
        """
        Wait for task completion by monitoring tmux session

        Returns the result dictionary from the task output
        """
        start_time = time.time()
        result_file = Path(f"/tmp/task-{task.task_id}-result.json")

        # Tell the script to write results to this file
        task.parameters['result_file'] = str(result_file)

        while time.time() - start_time < timeout:
            # Check if session is still alive
            try:
                subprocess.run(
                    ["tmux", "has-session", "-t", session_name],
                    check=True,
                    capture_output=True
                )
                # Session still alive, keep waiting
                time.sleep(5)

            except subprocess.CalledProcessError:
                # Session ended, check for results
                logger.info(f"Tmux session {session_name} ended")
                break

        # Read results if available
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
            result_file.unlink()  # Clean up
            return result

        # If no result file, try to capture output from tmux pane
        return self._capture_tmux_output(session_name)

    def _capture_tmux_output(self, session_name: str) -> Optional[Dict]:
        """Capture output from tmux pane"""
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"],
                check=True,
                capture_output=True,
                text=True
            )
            output = result.stdout

            # Try to parse as JSON if it looks like JSON
            if output.strip().startswith('{'):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    pass

            return {'output': output}

        except subprocess.CalledProcessError:
            return None

    def _format_parameters(self, parameters: Dict) -> str:
        """Format parameters as command line arguments"""
        args = []
        for key, value in parameters.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key}={value}")
        return " ".join(args)

    def _send_results_to_management(self, task: Task):
        """Send task results back to management Mac"""
        # Get management server URL from config
        management_host = self.config.management_host
        management_port = self.config.management_port

        # If management_host is 0.0.0.0, we need actual IP
        # For now, assume it's configured properly or use localhost
        if management_host == "0.0.0.0":
            management_host = "localhost"

        url = f"http://{management_host}:{management_port}/tasks/{task.task_id}/result"

        payload = {
            'task_id': task.task_id,
            'status': task.status.value,
            'result': task.result,
            'metrics': task.metrics,
            'resource_usage': task.resource_usage,
            'error': task.error
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Sent results for task {task.task_id} to management")
            else:
                logger.error(f"Failed to send results: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending results to management: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 5001):
        """Start the worker agent server"""
        logger.info(f"Starting worker agent '{self.worker_id}' on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)
