#!/usr/bin/env python3
"""
Example workflow demonstrating the complete multi-Mac AI workflow:
1. Create multiple ML training tasks with different hyperparameters
2. Distribute them across worker Macs
3. Monitor progress
4. Select best model based on accuracy
5. Clean up other worktrees and models
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.management.coordinator import TaskCoordinator
from src.common.task import TaskType


def run_hyperparameter_sweep():
    """Run a hyperparameter sweep across multiple workers"""

    # Initialize coordinator
    coordinator = TaskCoordinator(config_path='config.yaml')

    print("=" * 60)
    print("Multi-Mac Hyperparameter Sweep Example")
    print("=" * 60)

    # Define hyperparameter combinations to try
    hyperparameters = [
        {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10},
        {'learning_rate': 0.01, 'batch_size': 32, 'epochs': 10},
        {'learning_rate': 0.001, 'batch_size': 64, 'epochs': 10},
        {'learning_rate': 0.01, 'batch_size': 64, 'epochs': 10},
        {'learning_rate': 0.005, 'batch_size': 32, 'epochs': 15},
    ]

    # Create tasks
    task_ids = []
    for i, params in enumerate(hyperparameters):
        task = coordinator.create_task(
            task_type=TaskType.ML_TRAINING,
            name=f"model-lr{params['learning_rate']}-bs{params['batch_size']}-ep{params['epochs']}",
            description=f"Training with LR={params['learning_rate']}, BS={params['batch_size']}",
            script_path="examples/ml_training_example.py",
            parameters={
                'learning-rate': params['learning_rate'],
                'batch-size': params['batch_size'],
                'epochs': params['epochs'],
                'model-name': f"model-{i}"
            }
        )
        task_ids.append(task.task_id)
        print(f"Created task {i + 1}/{len(hyperparameters)}: {task.name}")

    # Assign tasks to workers
    print("\nAssigning tasks to workers...")
    for task_id in task_ids:
        worker_id = coordinator.assign_task(task_id)
        if worker_id:
            print(f"  Task {task_id[:8]} -> Worker {worker_id}")
        else:
            print(f"  Task {task_id[:8]} -> Waiting for available worker...")

    # Wait for completion
    print("\nWaiting for tasks to complete...")
    while True:
        completed = coordinator.get_completed_tasks(task_type=TaskType.ML_TRAINING)
        if len(completed) == len(task_ids):
            break
        print(f"  Progress: {len(completed)}/{len(task_ids)} tasks completed")
        time.sleep(10)

    print("\nAll tasks completed!")

    # Select best model
    print("\nAnalyzing results...")
    best_task = coordinator.select_best_task(
        completed,
        metric_key='accuracy',
        higher_is_better=True
    )

    if best_task:
        print(f"\nBest model: {best_task.name}")
        print(f"  Accuracy: {best_task.metrics.get('accuracy', 0):.4f}")
        print(f"  Loss: {best_task.metrics.get('loss', 0):.4f}")
        print(f"  Worktree: {best_task.worktree_name}")
        print(f"  Model: {best_task.result.get('model_path', 'N/A')}")

        # Display resource usage
        if best_task.resource_usage:
            print(f"\nResource Usage:")
            print(f"  CPU (avg): {best_task.resource_usage.get('cpu', {}).get('avg', 0):.1f}%")
            print(f"  Memory (avg): {best_task.resource_usage.get('memory', {}).get('avg', 0):.1f}%")

        # Clean up other models and worktrees
        print("\nCleaning up other models and worktrees...")
        coordinator.cleanup_except_best(
            completed,
            best_task,
            cleanup_worktrees=True,
            cleanup_models=True
        )

        print(f"\nKept best model: {best_task.result.get('model_path')}")
        print(f"Kept worktree: {best_task.worktree_name}")

    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    try:
        run_hyperparameter_sweep()
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
