#!/usr/bin/env python3
"""
Complete workflow with .mat data loading and ONNX export

This example demonstrates:
1. Generating .mat training data
2. Creating multiple ML training tasks with different hyperparameters
3. Each task trains on .mat data and exports to ONNX
4. Distributing tasks across worker Macs
5. Monitoring progress
6. Selecting the best model based on accuracy
7. Cleaning up other worktrees and models (keeping only the best ONNX model)
"""
import sys
import time
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.management.coordinator import TaskCoordinator
from src.common.task import TaskType


def generate_training_data():
    """Generate .mat training data"""
    print("=" * 60)
    print("Step 1: Generating .mat training data")
    print("=" * 60)

    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    # Check if data already exists
    train_file = data_dir / 'classification_train.mat'
    test_file = data_dir / 'classification_test.mat'

    if train_file.exists() and test_file.exists():
        print("  ✓ Training data already exists")
        return str(train_file), str(test_file)

    # Generate data
    cmd = [
        sys.executable,
        'examples/generate_mat_data.py',
        '--type', 'classification',
        '--n-samples', '2000',
        '--n-features', '20',
        '--n-classes', '3',
        '--output-dir', './data'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating data: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    return str(train_file), str(test_file)


def run_hyperparameter_sweep():
    """Run a hyperparameter sweep with .mat data and ONNX export"""

    # Generate training data
    train_file, test_file = generate_training_data()

    # Initialize coordinator
    coordinator = TaskCoordinator(config_path='config.yaml')

    print("\n" + "=" * 60)
    print("Step 2: Creating ML training tasks")
    print("=" * 60)

    # Define hyperparameter combinations to try
    hyperparameters = [
        {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 15,
            'hidden_sizes': '64,32',
            'dropout': 0.2
        },
        {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 15,
            'hidden_sizes': '64,32',
            'dropout': 0.2
        },
        {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 15,
            'hidden_sizes': '128,64',
            'dropout': 0.3
        },
        {
            'learning_rate': 0.005,
            'batch_size': 32,
            'epochs': 20,
            'hidden_sizes': '64,32,16',
            'dropout': 0.25
        },
        {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 15,
            'hidden_sizes': '64,32',
            'dropout': 0.1
        },
    ]

    # Create tasks
    task_ids = []
    for i, params in enumerate(hyperparameters):
        model_name = (f"model-lr{params['learning_rate']}-"
                     f"bs{params['batch_size']}-"
                     f"ep{params['epochs']}")

        task = coordinator.create_task(
            task_type=TaskType.ML_TRAINING,
            name=model_name,
            description=(f"Training with LR={params['learning_rate']}, "
                        f"BS={params['batch_size']}, "
                        f"Hidden={params['hidden_sizes']}"),
            script_path="examples/ml_training_mat_onnx.py",
            parameters={
                'train-data': train_file,
                'test-data': test_file,
                'task-type': 'classification',
                'learning-rate': params['learning_rate'],
                'batch-size': params['batch_size'],
                'epochs': params['epochs'],
                'hidden-sizes': params['hidden_sizes'],
                'dropout': params['dropout'],
                'model-name': model_name
            }
        )
        task_ids.append(task.task_id)
        print(f"  ✓ Created task {i + 1}/{len(hyperparameters)}: {task.name}")

    # Assign tasks to workers
    print("\n" + "=" * 60)
    print("Step 3: Assigning tasks to workers")
    print("=" * 60)

    for task_id in task_ids:
        worker_id = coordinator.assign_task(task_id)
        if worker_id:
            print(f"  Task {task_id[:8]} -> Worker {worker_id}")
        else:
            print(f"  Task {task_id[:8]} -> Queued (no workers available)")

    # Wait for completion
    print("\n" + "=" * 60)
    print("Step 4: Waiting for tasks to complete")
    print("=" * 60)

    while True:
        completed = coordinator.get_completed_tasks(task_type=TaskType.ML_TRAINING)
        if len(completed) == len(task_ids):
            break

        # Show progress
        completed_count = len(completed)
        running_count = len([t for t in coordinator.tasks.values()
                           if t.status.value == 'running'])
        pending_count = len(task_ids) - completed_count - running_count

        print(f"  Progress: {completed_count} completed, "
              f"{running_count} running, "
              f"{pending_count} pending")

        time.sleep(10)

    print("\n  ✓ All tasks completed!")

    # Analyze results
    print("\n" + "=" * 60)
    print("Step 5: Analyzing results")
    print("=" * 60)

    completed = coordinator.get_completed_tasks(task_type=TaskType.ML_TRAINING)

    # Display all results
    print("\nAll models:")
    for task in completed:
        accuracy = task.metrics.get('best_accuracy', task.metrics.get('final_accuracy', 0))
        loss = task.metrics.get('best_loss', task.metrics.get('final_test_loss', 0))
        onnx_success = task.result.get('onnx_export_success', False) if task.result else False

        print(f"  {task.name}")
        print(f"    Accuracy: {accuracy:.2f}%")
        print(f"    Loss: {loss:.4f}")
        print(f"    ONNX Export: {'✓' if onnx_success else '✗'}")
        if task.result and 'onnx_path' in task.result:
            print(f"    ONNX Path: {task.result['onnx_path']}")

    # Select best model by accuracy
    best_task = coordinator.select_best_task(
        completed,
        metric_key='best_accuracy',
        higher_is_better=True
    )

    if not best_task:
        # Try final_accuracy if best_accuracy not available
        best_task = coordinator.select_best_task(
            completed,
            metric_key='final_accuracy',
            higher_is_better=True
        )

    if best_task:
        print("\n" + "=" * 60)
        print("Step 6: Best model selected")
        print("=" * 60)

        accuracy = best_task.metrics.get('best_accuracy',
                                        best_task.metrics.get('final_accuracy', 0))
        loss = best_task.metrics.get('best_loss',
                                     best_task.metrics.get('final_test_loss', 0))

        print(f"\n  Model: {best_task.name}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Loss: {loss:.4f}")
        print(f"  Worktree: {best_task.worktree_name}")

        if best_task.result:
            print(f"  PyTorch Model: {best_task.result.get('model_path', 'N/A')}")
            print(f"  ONNX Model: {best_task.result.get('onnx_path', 'N/A')}")

        # Display resource usage
        if best_task.resource_usage:
            print(f"\n  Resource Usage:")
            cpu = best_task.resource_usage.get('cpu', {})
            memory = best_task.resource_usage.get('memory', {})
            print(f"    CPU (avg): {cpu.get('avg', 0):.1f}%")
            print(f"    Memory (avg): {memory.get('avg', 0):.1f}%")

            gpu = best_task.resource_usage.get('gpu', {})
            if gpu:
                for gpu_id, stats in gpu.items():
                    print(f"    GPU {gpu_id} Load (avg): {stats.get('load_avg', 0):.1f}%")
                    print(f"    GPU {gpu_id} Memory (avg): {stats.get('memory_avg', 0):.1f}%")

        # Clean up other models and worktrees
        print("\n" + "=" * 60)
        print("Step 7: Cleaning up non-optimal models")
        print("=" * 60)

        coordinator.cleanup_except_best(
            completed,
            best_task,
            cleanup_worktrees=True,
            cleanup_models=True
        )

        print(f"\n  ✓ Kept best ONNX model: {best_task.result.get('onnx_path')}")
        print(f"  ✓ Kept worktree: {best_task.worktree_name}")
        print(f"  ✓ Cleaned up {len(completed) - 1} other models")

    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)

    if best_task and best_task.result:
        print(f"\nBest model exported to: {best_task.result.get('onnx_path')}")
        print("You can now use this ONNX model for inference in any framework!")


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
