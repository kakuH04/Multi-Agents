#!/usr/bin/env python3
"""
Advanced ML Training Task with .mat data loading and ONNX export

This script demonstrates:
1. Loading training data from .mat files
2. Training a PyTorch neural network
3. Exporting the trained model to ONNX format
4. Reporting results back to the management system
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import onnx
import onnxruntime


class NeuralNetwork(nn.Module):
    """Simple feedforward neural network"""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout: float = 0.2):
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_mat_data(mat_file: str) -> tuple:
    """
    Load training data from .mat file

    Args:
        mat_file: Path to .mat file

    Returns:
        Tuple of (X, y) as numpy arrays
    """
    print(f"Loading data from {mat_file}...")

    mat_data = loadmat(mat_file)

    # Extract X and y from .mat file
    # Handle different possible key names
    if 'X' in mat_data:
        X = mat_data['X']
    elif 'data' in mat_data:
        X = mat_data['data']
    elif 'features' in mat_data:
        X = mat_data['features']
    else:
        raise ValueError(f"Could not find feature matrix in .mat file. Available keys: {list(mat_data.keys())}")

    if 'y' in mat_data:
        y = mat_data['y']
    elif 'labels' in mat_data:
        y = mat_data['labels']
    elif 'target' in mat_data:
        y = mat_data['target']
    else:
        raise ValueError(f"Could not find labels in .mat file. Available keys: {list(mat_data.keys())}")

    # Flatten if needed
    if len(y.shape) > 1:
        y = y.flatten()

    print(f"  Loaded X: {X.shape}, y: {y.shape}")

    return X, y


def train_model(
    train_file: str,
    test_file: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    hidden_sizes: list,
    dropout: float,
    task_type: str = 'classification'
) -> tuple:
    """
    Train a neural network model

    Args:
        train_file: Path to training .mat file
        test_file: Path to test .mat file
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        hidden_sizes: List of hidden layer sizes
        dropout: Dropout rate
        task_type: 'classification' or 'regression'

    Returns:
        Tuple of (model, metrics)
    """
    # Load data
    X_train, y_train = load_mat_data(train_file)
    X_test, y_test = load_mat_data(test_file)

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)

    if task_type == 'classification':
        # Determine number of classes
        n_classes = int(y_train.max()) + 1
        y_train_t = torch.LongTensor(y_train)
        y_test_t = torch.LongTensor(y_test)
        output_size = n_classes
        print(f"Classification task with {n_classes} classes")
    else:
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
        y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)
        output_size = 1
        print("Regression task")

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size, hidden_sizes, output_size, dropout)

    # Loss function and optimizer
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_accuracy = 0
    best_loss = float('inf')
    train_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

                if task_type == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if task_type == 'classification':
            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            best_accuracy = max(best_accuracy, accuracy)
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
        else:
            best_loss = min(best_loss, test_loss)
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}")

    # Prepare metrics
    metrics = {
        'final_train_loss': float(train_losses[-1]),
        'final_test_loss': float(test_losses[-1]),
        'best_loss': float(best_loss),
        'epochs_trained': epochs
    }

    if task_type == 'classification':
        metrics['final_accuracy'] = float(accuracies[-1])
        metrics['best_accuracy'] = float(best_accuracy)

    return model, metrics


def export_to_onnx(
    model: nn.Module,
    input_size: int,
    onnx_path: str,
    verify: bool = True
) -> bool:
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model
        input_size: Input feature size
        onnx_path: Output path for ONNX model
        verify: Whether to verify the exported model

    Returns:
        True if export successful
    """
    print(f"\nExporting model to ONNX: {onnx_path}")

    # Create dummy input
    dummy_input = torch.randn(1, input_size)

    # Export to ONNX
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"  ✓ Model exported to {onnx_path}")

    # Verify the model
    if verify:
        try:
            # Load and check the ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ✓ ONNX model verified successfully")

            # Test with ONNX Runtime
            ort_session = onnxruntime.InferenceSession(onnx_path)

            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare with PyTorch output
            with torch.no_grad():
                pytorch_output = model(dummy_input).numpy()

            # Check if outputs match
            np.testing.assert_allclose(pytorch_output, ort_outputs[0], rtol=1e-3, atol=1e-5)
            print(f"  ✓ ONNX Runtime inference matches PyTorch")

            return True

        except Exception as e:
            print(f"  ✗ ONNX verification failed: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description='ML Training with .mat data and ONNX export')
    parser.add_argument('--train-data', required=True, help='Path to training .mat file')
    parser.add_argument('--test-data', required=True, help='Path to test .mat file')
    parser.add_argument('--task-type', choices=['classification', 'regression'],
                       default='classification', help='Task type')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden-sizes', default='64,32', help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--model-name', default='neural-network', help='Model name')
    parser.add_argument('--result-file', help='Path to save results JSON')

    args = parser.parse_args()

    try:
        # Parse hidden sizes
        hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
        print(f"Hidden layer sizes: {hidden_sizes}")

        # Train model
        model, metrics = train_model(
            train_file=args.train_data,
            test_file=args.test_data,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
            task_type=args.task_type
        )

        # Create output directories
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)

        onnx_dir = Path('./models/onnx')
        onnx_dir.mkdir(exist_ok=True)

        # Save PyTorch model
        pytorch_path = model_dir / f"{args.model_name}.pth"
        torch.save(model.state_dict(), pytorch_path)
        print(f"\n✓ PyTorch model saved: {pytorch_path}")

        # Export to ONNX
        onnx_path = onnx_dir / f"{args.model_name}.onnx"

        # Get input size from first layer
        input_size = model.network[0].in_features

        onnx_success = export_to_onnx(model, input_size, str(onnx_path), verify=True)

        # Prepare result
        result = {
            'status': 'success',
            'model_path': str(pytorch_path),
            'onnx_path': str(onnx_path),
            'onnx_export_success': onnx_success,
            'model_name': args.model_name,
            'metrics': metrics,
            'parameters': {
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'hidden_sizes': hidden_sizes,
                'dropout': args.dropout,
                'task_type': args.task_type
            }
        }

        # Write result to file if specified
        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(result, f, indent=2)

        # Print result as JSON for tmux capture
        print("\n" + "=" * 60)
        print("RESULT:")
        print(json.dumps(result, indent=2))
        print("=" * 60)

        return 0

    except Exception as e:
        import traceback
        error_result = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(error_result, f, indent=2)

        print(json.dumps(error_result, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())
