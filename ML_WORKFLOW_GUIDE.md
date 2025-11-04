# ML Training Workflow Guide

Complete guide for running ML training tasks with .mat data and ONNX export.

## Overview

The Multi-Mac AI Workflow system supports advanced ML training workflows with:
- **MATLAB .mat file loading** - Load training data from .mat files
- **PyTorch neural networks** - Train customizable neural networks
- **ONNX export** - Export trained models to ONNX format for cross-platform deployment
- **Automatic hyperparameter tuning** - Run multiple experiments in parallel
- **Resource monitoring** - Track CPU, GPU, Memory usage
- **Best model selection** - Automatically select the best performing model

## Quick Start

### 1. Generate Training Data

Create .mat training data:

```bash
python examples/generate_mat_data.py \
  --type classification \
  --n-samples 2000 \
  --n-features 20 \
  --n-classes 3 \
  --output-dir ./data
```

This creates:
- `data/classification_train.mat` - Training dataset
- `data/classification_test.mat` - Test dataset

### 2. Train a Single Model

Train with .mat data and export to ONNX:

```bash
python examples/ml_training_mat_onnx.py \
  --train-data data/classification_train.mat \
  --test-data data/classification_test.mat \
  --task-type classification \
  --epochs 20 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --hidden-sizes 64,32 \
  --dropout 0.2 \
  --model-name my-model
```

Output:
- `models/my-model.pth` - PyTorch model
- `models/onnx/my-model.onnx` - ONNX model

### 3. Run Hyperparameter Sweep

Run multiple training experiments in parallel:

```bash
python examples/run_workflow_mat_onnx.py
```

This will:
1. Generate .mat training data (if not exists)
2. Create 5 training tasks with different hyperparameters
3. Distribute across available workers
4. Train all models in parallel
5. Export each to ONNX
6. Select the best model based on accuracy
7. Clean up non-optimal models

## Data Formats

### .mat File Structure

Training and test .mat files should contain:

```matlab
% Classification data
X       % Features matrix (n_samples × n_features)
y       % Labels vector (n_samples × 1)

% Regression data
X       % Features matrix (n_samples × n_features)
y       % Target values (n_samples × 1)
```

Alternative key names are supported:
- Features: `X`, `data`, `features`
- Labels: `y`, `labels`, `target`

### Generating Custom Data

Use the data generator with custom parameters:

```bash
# Classification
python examples/generate_mat_data.py \
  --type classification \
  --n-samples 5000 \
  --n-features 50 \
  --n-classes 5 \
  --noise 0.1 \
  --train-ratio 0.8

# Regression
python examples/generate_mat_data.py \
  --type regression \
  --n-samples 3000 \
  --n-features 30 \
  --noise 0.05
```

## Model Architecture

The neural network is fully customizable:

```bash
# Simple network: input -> 64 -> output
--hidden-sizes 64

# Deeper network: input -> 128 -> 64 -> 32 -> output
--hidden-sizes 128,64,32

# Wide network: input -> 256 -> 128 -> output
--hidden-sizes 256,128
```

Each hidden layer includes:
- Linear transformation
- ReLU activation
- Dropout regularization

## Training Parameters

### Key Parameters

- `--epochs` - Number of training epochs (default: 10)
- `--learning-rate` - Adam optimizer learning rate (default: 0.001)
- `--batch-size` - Mini-batch size (default: 32)
- `--hidden-sizes` - Hidden layer sizes, comma-separated (default: 64,32)
- `--dropout` - Dropout rate for regularization (default: 0.2)

### Task Types

**Classification:**
```bash
--task-type classification
```
- Uses CrossEntropyLoss
- Reports accuracy and loss
- Multi-class support

**Regression:**
```bash
--task-type regression
```
- Uses MSE loss
- Reports loss metrics

## ONNX Export

Every trained model is automatically exported to ONNX format with:

- **Opset version 11** - Compatible with most frameworks
- **Dynamic batch size** - Support variable batch sizes
- **Verification** - Automatic verification against PyTorch
- **Runtime testing** - Tests with ONNX Runtime

### Using ONNX Models

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('models/onnx/my-model.onnx')

# Prepare input
input_data = np.random.randn(1, 20).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
predictions = outputs[0]
```

### ONNX Deployment

ONNX models can be deployed to:
- **Mobile** - iOS (Core ML), Android (NNAPI)
- **Web** - ONNX.js in browsers
- **Edge** - ONNX Runtime on IoT devices
- **Cloud** - Azure ML, AWS SageMaker, GCP AI Platform
- **Other frameworks** - TensorFlow, TensorRT, OpenVINO

## Distributed Training Workflow

### Using the Client

Submit training tasks to the management server:

```bash
python client.py create \
  --type ml_training \
  --name "experiment-1" \
  --script "examples/ml_training_mat_onnx.py" \
  --params '{
    "train-data": "data/classification_train.mat",
    "test-data": "data/classification_test.mat",
    "task-type": "classification",
    "epochs": 20,
    "learning-rate": 0.001,
    "batch-size": 32,
    "hidden-sizes": "128,64",
    "dropout": 0.3,
    "model-name": "experiment-1"
  }'
```

### Monitoring Tasks

```bash
# List all tasks
python client.py list

# Get specific task
python client.py get <task-id>

# View training in tmux
tmux ls
tmux attach -t ml-task-<task-id>
```

## Example Workflows

### 1. Learning Rate Search

```python
learning_rates = [0.0001, 0.001, 0.01, 0.1]

for lr in learning_rates:
    create_task(
        parameters={
            'learning-rate': lr,
            'epochs': 20,
            # ... other params
        }
    )
```

### 2. Architecture Search

```python
architectures = [
    '32',
    '64,32',
    '128,64',
    '128,64,32',
    '256,128,64'
]

for arch in architectures:
    create_task(
        parameters={
            'hidden-sizes': arch,
            # ... other params
        }
    )
```

### 3. Regularization Search

```python
import itertools

batch_sizes = [16, 32, 64]
dropout_rates = [0.1, 0.2, 0.3]

for bs, dr in itertools.product(batch_sizes, dropout_rates):
    create_task(
        parameters={
            'batch-size': bs,
            'dropout': dr,
            # ... other params
        }
    )
```

## Result Analysis

After training, analyze results:

```python
from src.management.coordinator import TaskCoordinator

coordinator = TaskCoordinator()

# Get completed tasks
completed = coordinator.get_completed_tasks()

# Find best by accuracy
best_task = coordinator.select_best_task(
    completed,
    metric_key='best_accuracy',
    higher_is_better=True
)

# Access results
print(f"Best accuracy: {best_task.metrics['best_accuracy']:.2f}%")
print(f"ONNX model: {best_task.result['onnx_path']}")

# Resource usage
print(f"Avg CPU: {best_task.resource_usage['cpu']['avg']:.1f}%")
print(f"Avg GPU: {best_task.resource_usage['gpu'][0]['load_avg']:.1f}%")
```

## Performance Tips

### GPU Acceleration

Ensure PyTorch uses GPU:
```python
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

Models automatically use GPU if available.

### Parallel Training

- **Multiple workers** - Distribute across Macs
- **Concurrent tasks** - Run multiple on same Mac (CPU/GPU dependent)
- **Batch size** - Larger batches for GPU, smaller for CPU

### Memory Management

Monitor memory usage:
```bash
# During training, check:
# - Management server logs
# - Worker resource monitoring
# - Task result metrics
```

## Troubleshooting

### .mat File Errors

**Error: Could not find feature matrix**
- Check .mat file contains `X`, `data`, or `features` key
- Verify with: `scipy.io.loadmat('file.mat').keys()`

**Error: Shape mismatch**
- Ensure X is 2D: (n_samples, n_features)
- Ensure y is 1D: (n_samples,)

### ONNX Export Errors

**Error: ONNX export failed**
- Check PyTorch version compatibility
- Verify model forward pass works
- Try older opset version

**Error: ONNX verification failed**
- May be numerical precision differences (usually OK)
- Check model output manually

### Training Issues

**Loss not decreasing:**
- Reduce learning rate
- Increase model capacity
- Check data normalization
- Increase epochs

**Overfitting:**
- Increase dropout
- Reduce model size
- Add more training data
- Reduce epochs

## Best Practices

1. **Start Small** - Test with small dataset first
2. **Validate Data** - Check .mat files load correctly
3. **Monitor Resources** - Watch GPU/CPU usage
4. **Save Checkpoints** - Models saved automatically
5. **Version Experiments** - Use descriptive model names
6. **Clean Up** - Remove non-optimal models regularly

## Advanced Topics

### Custom Loss Functions

Modify `ml_training_mat_onnx.py` to add custom losses:

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Your custom loss
        return loss

criterion = CustomLoss()
```

### Transfer Learning

Load pre-trained weights:

```python
model = NeuralNetwork(...)
model.load_state_dict(torch.load('pretrained.pth'))
# Fine-tune on new data
```

### Early Stopping

Add early stopping logic:

```python
best_loss = float('inf')
patience = 5
no_improve = 0

for epoch in range(epochs):
    # ... training ...
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        break
```

## Next Steps

- Explore advanced architectures (CNN, RNN, Transformers)
- Implement data augmentation
- Add learning rate scheduling
- Try different optimizers (SGD, AdamW)
- Implement cross-validation
- Add model ensembling

For more information, see the main [README.md](README.md).
