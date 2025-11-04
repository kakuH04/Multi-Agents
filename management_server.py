#!/usr/bin/env python3
"""
Management Mac Server
Run this on the Mac that will coordinate tasks
"""
import argparse
import sys
from src.management.coordinator import TaskCoordinator


def main():
    parser = argparse.ArgumentParser(description='Multi-Mac Management Server')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Mac AI Workflow - Management Server")
    print("=" * 60)

    try:
        coordinator = TaskCoordinator(config_path=args.config)
        coordinator.run()
    except KeyboardInterrupt:
        print("\nShutting down management server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
