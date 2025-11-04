# Multi-Mac AI Workflow System

A distributed task execution system for running ML training and coding tasks across multiple Macs, with automatic resource monitoring, git worktree management, and intelligent result selection.

## Features

- **Distributed Task Execution**: Run multiple tasks in parallel across different Macs
- **Management/Worker Architecture**: One Mac coordinates while others execute tasks
- **Git Worktree Integration**: Each task runs in its own isolated git worktree
- **Tmux Session Management**: Tasks run in persistent tmux sessions
- **Resource Monitoring**: Track CPU, GPU, Memory, and Disk usage during execution
- **Automatic Result Selection**: Choose the best result based on custom metrics
- **Intelligent Cleanup**: Automatically remove non-optimal worktrees and models
- **REST API**: Simple HTTP API for task submission and monitoring
- **ML Training Support**: Load .mat data, train PyTorch models, export to ONNX
- **Hyperparameter Tuning**: Run multiple experiments in parallel with auto-selection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Management Mac                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Task Coordinator                                     │   │
│  │  - Task queue management                              │   │
│  │  - Worker assignment                                  │   │
│  │  - Result collection                                  │   │
│  │  - Best result selection                              │   │
│  │  - Cleanup orchestration                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────┬────────────────────────────┬───────────────┘
                  │                            │
          HTTP API│                            │HTTP API
                  │                            │
    ┌─────────────▼──────────┐    ┌───────────▼────────────┐
    │    Worker Mac 1        │    │    Worker Mac 2        │
    │  ┌──────────────────┐  │    │  ┌──────────────────┐  │
    │  │  Worker Agent    │  │    │  │  Worker Agent    │  │
    │  │  - Task executor │  │    │  │  - Task executor │  │
    │  │  - Tmux manager  │  │    │  │  - Tmux manager  │  │
    │  │  - Git worktree  │  │    │  │  - Git worktree  │  │
    │  │  - Resource mon. │  │    │  │  - Resource mon. │  │
    │  └──────────────────┘  │    │  └──────────────────┘  │
    └────────────────────────┘    └────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Git 2.5+ (for worktree support)
- tmux
- SSH access between Macs (for distributed setup)
- Optional: CUDA-compatible GPU for accelerated training

### Setup

1. Clone the repository on all Macs:
```bash
git clone <your-repo-url>
cd Multi-Agents
```

2. Install dependencies on all Macs:
```bash
pip install -r requirements.txt
```

3. Configure the system by editing `config.yaml`:
```yaml
management:
  host: "0.0.0.0"
  port: 5000

workers:
  - name: "worker-1"
    host: "192.168.1.100"  # IP of worker Mac
    port: 5001
    ssh_user: "username"
    max_concurrent_tasks: 3

  - name: "worker-2"
    host: "192.168.1.101"
    port: 5001
    ssh_user: "username"
    max_concurrent_tasks: 3
```

4. Make scripts executable:
```bash
chmod +x management_server.py worker_server.py client.py
chmod +x examples/*.py
```

## Usage

### Starting the System

**On Management Mac:**
```bash
python management_server.py --config config.yaml
```

**On Each Worker Mac:**
```bash
python worker_server.py --worker-id worker-1 --port 5001
```

### Submitting Tasks

Using the client CLI:

```bash
# Create an ML training task
python client.py create \
  --type ml_training \
  --name "training-experiment-1" \
  --description "Test hyperparameters" \
  --script "examples/ml_training_example.py" \
  --params '{"learning-rate": 0.001, "epochs": 10, "batch-size": 32}'

# List all tasks
python client.py list

# Get task status
python client.py get <task-id>
```

Using the REST API:

```bash
# Create a task
curl -X POST http://localhost:5000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "ml_training",
    "name": "my-training-task",
    "description": "Train a model",
    "script_path": "examples/ml_training_example.py",
    "parameters": {
      "learning-rate": 0.001,
      "epochs": 10,
      "batch-size": 32
    }
  }'

# Get task status
curl http://localhost:5000/tasks/<task-id>

# List all tasks
curl http://localhost:5000/tasks
```

### Running a Complete Workflow

See `examples/run_workflow_example.py` for a complete example that:
1. Creates multiple training tasks with different hyperparameters
2. Distributes them across workers
3. Monitors progress
4. Selects the best model
5. Cleans up other worktrees and models

```bash
python examples/run_workflow_example.py
```

## Writing Custom Tasks

Tasks are Python scripts that:
1. Accept parameters via command-line arguments
2. Perform work (training, coding, testing, etc.)
3. Write results to a JSON file
4. Print results for tmux capture

### Example ML Training Task

```python
#!/usr/bin/env python3
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--result-file', help='Result output file')
    args = parser.parse_args()

    # Your training code here
    metrics = train_model(args.epochs, args.learning_rate)

    # Prepare result
    result = {
        'status': 'success',
        'metrics': metrics,
        'model_path': './models/my_model.pth'
    }

    # Write result
    if args.result_file:
        with open(args.result_file, 'w') as f:
            json.dump(result, f)

    print(json.dumps(result))

if __name__ == '__main__':
    main()
```

## Git Worktree Workflow

Each task runs in its own git worktree:

1. **Task Creation**: System creates a new worktree with branch `task/<name>`
2. **Task Execution**: All changes happen in isolated worktree
3. **Completion**: Changes are committed to the worktree's branch
4. **Selection**: Best result is identified based on metrics
5. **Cleanup**: Non-optimal worktrees are removed

Example worktree structure:
```
project/
├── .git/                    # Main repo
├── src/                     # Main working directory
├── worktrees/
│   ├── task-experiment-1/   # Worktree for task 1
│   ├── task-experiment-2/   # Worktree for task 2
│   └── task-experiment-3/   # Worktree for task 3 (best - kept)
```

## Resource Monitoring

The system automatically monitors:
- **CPU Usage**: Average, max, min percentage
- **Memory Usage**: Average, max, min percentage
- **GPU Usage**: Load, memory, temperature (if available)
- **Disk Usage**: Space utilization

Metrics are collected at configurable intervals and included in task results.

## Selecting Best Results

After tasks complete, use metrics to select the best:

```python
from src.management.coordinator import TaskCoordinator

coordinator = TaskCoordinator()

# Get completed tasks
completed = coordinator.get_completed_tasks()

# Select best by accuracy (higher is better)
best = coordinator.select_best_task(
    completed,
    metric_key='accuracy',
    higher_is_better=True
)

# Clean up others
coordinator.cleanup_except_best(
    completed,
    best,
    cleanup_worktrees=True,
    cleanup_models=True
)
```

## Configuration Reference

### Management Settings
- `management.host`: Host to bind management server (default: 0.0.0.0)
- `management.port`: Port for management server (default: 5000)
- `management.max_workers`: Maximum number of workers (default: 10)
- `management.result_dir`: Directory for results (default: ./results)

### Worker Settings
Each worker requires:
- `name`: Unique identifier
- `host`: IP address or hostname
- `port`: Port number (default: 5001)
- `ssh_user`: SSH username for remote access
- `max_concurrent_tasks`: Max parallel tasks (default: 3)

### Git Settings
- `git.worktree_base`: Base directory for worktrees (default: ./worktrees)
- `git.keep_best_n`: Number of best results to keep (default: 1)

### Monitoring Settings
- `monitoring.interval`: Sampling interval in seconds (default: 5)
- `monitoring.track_gpu`: Enable GPU monitoring (default: true)
- `monitoring.track_cpu`: Enable CPU monitoring (default: true)
- `monitoring.track_memory`: Enable memory monitoring (default: true)

### Tmux Settings
- `tmux.session_prefix`: Prefix for tmux sessions (default: ml-task)
- `tmux.window_name`: Window name (default: training)

## Troubleshooting

### Workers not connecting
- Check firewall settings
- Verify network connectivity: `ping <worker-ip>`
- Ensure worker server is running: `curl http://<worker-ip>:<port>/health`

### Tasks not starting
- Check tmux is installed: `tmux -V`
- Verify git worktree support: `git worktree --help`
- Check logs in worker output

### Resource monitoring issues
- GPU monitoring requires GPUtil: `pip install GPUtil`
- Check GPU availability: `python -c "import GPUtil; print(GPUtil.getGPUs())"`

### Worktree errors
- Ensure base directory is a git repository
- Clean up stale worktrees: `git worktree prune`

## API Reference

### Management Server API

**Create Task**
```
POST /tasks
Content-Type: application/json

{
  "task_type": "ml_training" | "coding" | "testing" | "data_processing",
  "name": "task-name",
  "description": "description",
  "script_path": "path/to/script.py",
  "parameters": {...}
}
```

**Get Task**
```
GET /tasks/<task-id>
```

**List Tasks**
```
GET /tasks
```

**Update Task Result** (used by workers)
```
POST /tasks/<task-id>/result
Content-Type: application/json

{
  "task_id": "...",
  "status": "completed" | "failed",
  "result": {...},
  "metrics": {...},
  "resource_usage": {...},
  "error": "error message if failed"
}
```

**Health Check**
```
GET /health
```

### Worker Server API

**Execute Task**
```
POST /execute
Content-Type: application/json

{task object}
```

**Get Task Status**
```
GET /tasks/<task-id>
```

**Health Check**
```
GET /health
```

## Examples

See the `examples/` directory for:

**Basic Examples:**
- `ml_training_example.py`: Simple ML training task template
- `coding_task_example.py`: Coding/testing task template
- `run_workflow_example.py`: Complete workflow example

**Advanced ML Examples:**
- `generate_mat_data.py`: Generate .mat training data
- `ml_training_mat_onnx.py`: ML training with .mat data and ONNX export
- `run_workflow_mat_onnx.py`: Complete hyperparameter sweep with ONNX

### ML Training with .mat Data and ONNX Export

The system supports advanced ML workflows:

```bash
# Generate training data
python examples/generate_mat_data.py \
  --type classification \
  --n-samples 2000 \
  --n-features 20 \
  --n-classes 3

# Train model and export to ONNX
python examples/ml_training_mat_onnx.py \
  --train-data data/classification_train.mat \
  --test-data data/classification_test.mat \
  --epochs 20 \
  --learning-rate 0.001 \
  --hidden-sizes 64,32

# Run complete hyperparameter sweep
python examples/run_workflow_mat_onnx.py
```

**Features:**
- Load training data from MATLAB .mat files
- Train PyTorch neural networks
- Automatic ONNX export for cross-platform deployment
- Hyperparameter search across multiple workers
- Best model selection and automatic cleanup

For detailed ML workflow documentation, see [ML_WORKFLOW_GUIDE.md](ML_WORKFLOW_GUIDE.md)

## License

MIT

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Support

For questions and support, please open an issue on GitHub.
