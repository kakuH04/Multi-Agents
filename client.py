#!/usr/bin/env python3
"""
Client for submitting tasks and managing the workflow
"""
import argparse
import requests
import json
import sys
import time
from typing import List, Dict


class WorkflowClient:
    """Client for interacting with the management server"""

    def __init__(self, management_url: str):
        self.management_url = management_url.rstrip('/')

    def create_task(
        self,
        task_type: str,
        name: str,
        description: str,
        script_path: str,
        parameters: Dict = None
    ) -> Dict:
        """Create a new task"""
        url = f"{self.management_url}/tasks"
        payload = {
            'task_type': task_type,
            'name': name,
            'description': description,
            'script_path': script_path,
            'parameters': parameters or {}
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_task(self, task_id: str) -> Dict:
        """Get task status"""
        url = f"{self.management_url}/tasks/{task_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def list_tasks(self) -> List[Dict]:
        """List all tasks"""
        url = f"{self.management_url}/tasks"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, task_ids: List[str], poll_interval: int = 5):
        """Wait for multiple tasks to complete"""
        pending = set(task_ids)

        while pending:
            for task_id in list(pending):
                task = self.get_task(task_id)
                status = task['status']

                if status in ['completed', 'failed', 'cancelled']:
                    print(f"Task {task_id[:8]} {status}")
                    pending.remove(task_id)

            if pending:
                time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description='Multi-Mac Workflow Client')
    parser.add_argument(
        '--management-url',
        default='http://localhost:5000',
        help='Management server URL (default: http://localhost:5000)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Create task command
    create_parser = subparsers.add_parser('create', help='Create a new task')
    create_parser.add_argument('--type', required=True, choices=['ml_training', 'coding', 'testing', 'data_processing'])
    create_parser.add_argument('--name', required=True, help='Task name')
    create_parser.add_argument('--description', default='', help='Task description')
    create_parser.add_argument('--script', required=True, help='Path to script to execute')
    create_parser.add_argument('--params', help='Parameters as JSON string')

    # List tasks command
    list_parser = subparsers.add_parser('list', help='List all tasks')

    # Get task command
    get_parser = subparsers.add_parser('get', help='Get task status')
    get_parser.add_argument('task_id', help='Task ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = WorkflowClient(args.management_url)

    try:
        if args.command == 'create':
            params = json.loads(args.params) if args.params else {}
            task = client.create_task(
                task_type=args.type,
                name=args.name,
                description=args.description,
                script_path=args.script,
                parameters=params
            )
            print(f"Created task: {task['task_id']}")
            print(json.dumps(task, indent=2))

        elif args.command == 'list':
            tasks = client.list_tasks()
            print(f"Total tasks: {len(tasks)}")
            for task in tasks:
                print(f"  {task['task_id'][:8]} - {task['name']} ({task['status']})")

        elif args.command == 'get':
            task = client.get_task(args.task_id)
            print(json.dumps(task, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
