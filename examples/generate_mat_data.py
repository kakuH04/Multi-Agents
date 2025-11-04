#!/usr/bin/env python3
"""
Generate sample .mat training data for ML experiments
Creates MATLAB-compatible .mat files with training and test datasets
"""
import numpy as np
from scipy.io import savemat
from pathlib import Path
import argparse


def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    noise: float = 0.1
) -> dict:
    """
    Generate synthetic classification dataset

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        noise: Noise level

    Returns:
        Dictionary with X (features) and y (labels)
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate labels with some structure
    # Create centroids for each class
    centroids = np.random.randn(n_classes, n_features) * 3

    # Assign samples to classes based on nearest centroid
    y = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        y[i] = np.argmin(distances)

    # Add noise
    X += np.random.randn(n_samples, n_features) * noise

    return {'X': X, 'y': y}


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,
    noise: float = 0.1
) -> dict:
    """
    Generate synthetic regression dataset

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level

    Returns:
        Dictionary with X (features) and y (targets)
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate targets with linear relationship + non-linear terms
    weights = np.random.randn(n_features)
    y = np.dot(X, weights)

    # Add non-linear terms
    y += 0.5 * np.sin(X[:, 0]) * np.cos(X[:, 1])

    # Add noise
    y += np.random.randn(n_samples) * noise

    return {'X': X, 'y': y}


def split_dataset(data: dict, train_ratio: float = 0.8) -> tuple:
    """
    Split dataset into training and test sets

    Args:
        data: Dictionary with X and y
        train_ratio: Ratio of training samples

    Returns:
        Tuple of (train_data, test_data)
    """
    n_samples = data['X'].shape[0]
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = {
        'X': data['X'][train_idx],
        'y': data['y'][train_idx]
    }

    test_data = {
        'X': data['X'][test_idx],
        'y': data['y'][test_idx]
    }

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Generate .mat training data')
    parser.add_argument('--type', choices=['classification', 'regression'],
                       default='classification', help='Dataset type')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples')
    parser.add_argument('--n-features', type=int, default=20,
                       help='Number of features')
    parser.add_argument('--n-classes', type=int, default=3,
                       help='Number of classes (classification only)')
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Noise level')
    parser.add_argument('--output-dir', default='./data',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.type} dataset...")
    print(f"  Samples: {args.n_samples}")
    print(f"  Features: {args.n_features}")

    # Generate data
    if args.type == 'classification':
        print(f"  Classes: {args.n_classes}")
        data = generate_classification_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_classes=args.n_classes,
            noise=args.noise
        )
    else:
        data = generate_regression_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise
        )

    # Split into train/test
    train_data, test_data = split_dataset(data, args.train_ratio)

    # Save as .mat files
    train_file = output_dir / f"{args.type}_train.mat"
    test_file = output_dir / f"{args.type}_test.mat"

    savemat(str(train_file), train_data)
    savemat(str(test_file), test_data)

    print(f"\nDataset generated successfully!")
    print(f"  Training data: {train_file} ({train_data['X'].shape[0]} samples)")
    print(f"  Test data:     {test_file} ({test_data['X'].shape[0]} samples)")
    print(f"  Feature shape: {train_data['X'].shape}")
    print(f"  Label shape:   {train_data['y'].shape}")


if __name__ == '__main__':
    main()
