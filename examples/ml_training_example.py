#!/usr/bin/env python3
"""
Example ML Training Task
This script demonstrates how to write a task that trains a model
and reports results back to the management system.
"""
import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path


def train_model(epochs: int, learning_rate: float, batch_size: int, model_name: str):
    """Simulate ML model training"""
    print(f"Starting training: {model_name}")
    print(f"Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}")

    # Simulate training with progress
    best_accuracy = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        # Simulate epoch training
        time.sleep(1)  # Simulate work

        # Generate fake metrics (in real scenario, these come from actual training)
        loss = 2.0 - (epoch / epochs) * 1.5 + np.random.random() * 0.1
        accuracy = 0.5 + (epoch / epochs) * 0.4 + np.random.random() * 0.05

        best_accuracy = max(best_accuracy, accuracy)
        best_loss = min(best_loss, loss)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return {
        'best_accuracy': float(best_accuracy),
        'best_loss': float(best_loss),
        'final_accuracy': float(accuracy),
        'final_loss': float(loss),
        'epochs_trained': epochs
    }


def main():
    parser = argparse.ArgumentParser(description='ML Training Example')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-name', default='example-model', help='Model name')
    parser.add_argument('--result-file', help='Path to save results JSON')

    args = parser.parse_args()

    try:
        # Train model
        metrics = train_model(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            model_name=args.model_name
        )

        # Save model (simulate)
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{args.model_name}.pth"

        # In real scenario, save actual model here
        with open(model_path, 'w') as f:
            f.write("Simulated model weights\n")

        # Prepare result
        result = {
            'status': 'success',
            'model_path': str(model_path),
            'model_name': args.model_name,
            'metrics': metrics,
            'parameters': {
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size
            }
        }

        # Write result to file if specified
        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(result, f, indent=2)

        # Also print as JSON for tmux capture
        print("\n" + "=" * 60)
        print("RESULT:")
        print(json.dumps(result, indent=2))
        print("=" * 60)

        return 0

    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e)
        }

        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(error_result, f, indent=2)

        print(json.dumps(error_result, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())
