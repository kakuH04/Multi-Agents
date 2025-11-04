# Quick Start Guide

Get up and running with the Multi-Mac AI Workflow system in 5 minutes.

## Single Mac Testing (Development)

For testing on a single Mac before deploying across multiple machines:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Management Server

In terminal 1:
```bash
python management_server.py
```

### 3. Start Worker Agent

In terminal 2:
```bash
python worker_server.py --worker-id local-worker --port 5001
```

### 4. Submit a Test Task

In terminal 3:
```bash
python client.py create \
  --type ml_training \
  --name "test-training" \
  --script "examples/ml_training_example.py" \
  --params '{"epochs": 5, "learning-rate": 0.001, "batch-size": 32}'
```

### 5. Monitor Progress

```bash
# List all tasks
python client.py list

# Get specific task status
python client.py get <task-id>
```

## Multi-Mac Setup

### On Management Mac:

```bash
# 1. Clone and install
git clone <repo-url>
cd Multi-Agents
pip install -r requirements.txt

# 2. Configure workers in config.yaml
# Edit config.yaml to add worker Mac IPs

# 3. Start management server
python management_server.py
```

### On Each Worker Mac:

```bash
# 1. Clone and install
git clone <repo-url>
cd Multi-Agents
pip install -r requirements.txt

# 2. Start worker
python worker_server.py --worker-id worker-1 --port 5001
```

## Running a Complete Workflow

Run the hyperparameter sweep example:

```bash
python examples/run_workflow_example.py
```

This will:
1. Create 5 training tasks with different hyperparameters
2. Distribute them across available workers
3. Monitor progress and collect results
4. Select the best model based on accuracy
5. Clean up non-optimal worktrees and models

## Viewing Task Execution

Tasks run in tmux sessions. To view a running task:

```bash
# List tmux sessions
tmux ls

# Attach to a session
tmux attach -t ml-task-<task-id>

# Detach: Press Ctrl+b then d
```

## Monitoring Resources

Resource usage is automatically collected and included in task results. View them:

```bash
python client.py get <task-id>
```

Look for the `resource_usage` section in the output.

## Checking Git Worktrees

```bash
# List all worktrees
git worktree list

# View files in a worktree
ls worktrees/task-<name>/

# Switch to a worktree
cd worktrees/task-<name>/
```

## ML Training with .mat Data and ONNX

Quick example of advanced ML workflow:

```bash
# 1. Generate training data
python examples/generate_mat_data.py \
  --type classification \
  --n-samples 1000 \
  --n-features 20 \
  --n-classes 3

# 2. Train a model (exports to ONNX automatically)
python examples/ml_training_mat_onnx.py \
  --train-data data/classification_train.mat \
  --test-data data/classification_test.mat \
  --epochs 15 \
  --learning-rate 0.001 \
  --hidden-sizes 64,32

# 3. Run full hyperparameter sweep across workers
python examples/run_workflow_mat_onnx.py
```

This will:
- Load .mat training data
- Train PyTorch neural networks
- Export to ONNX format
- Select best model automatically
- Clean up non-optimal models

See [ML_WORKFLOW_GUIDE.md](ML_WORKFLOW_GUIDE.md) for detailed ML documentation.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [examples/](examples/) for task templates
- See [ML_WORKFLOW_GUIDE.md](ML_WORKFLOW_GUIDE.md) for ML training guide
- Customize [config.yaml](config.yaml) for your setup
- Write your own task scripts following the examples

## Common Commands

```bash
# Management server
python management_server.py --config config.yaml

# Worker server
python worker_server.py --worker-id <id> --port <port>

# Create task
python client.py create --type <type> --name <name> --script <path> --params <json>

# List tasks
python client.py list

# Get task status
python client.py get <task-id>

# Health check
curl http://localhost:5000/health
curl http://localhost:5001/health
```

## Troubleshooting

**Workers not showing up:**
- Check `config.yaml` has correct worker IPs
- Verify network connectivity: `ping <worker-ip>`
- Check worker server is running: `curl http://<worker-ip>:5001/health`

**Tasks not starting:**
- Verify tmux is installed: `tmux -V`
- Check git worktree support: `git worktree list`
- Look at server logs for errors

**Import errors:**
- Ensure you're in the project root directory
- Install all requirements: `pip install -r requirements.txt`
