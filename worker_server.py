#!/usr/bin/env python3
"""
Worker Mac Server
Run this on each Mac that will execute tasks
"""
import argparse
import sys
from src.worker.agent import WorkerAgent


def main():
    parser = argparse.ArgumentParser(description='Multi-Mac Worker Server')
    parser.add_argument(
        '--worker-id',
        required=True,
        help='Unique identifier for this worker'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='Port to bind to (default: 5001)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Multi-Mac AI Workflow - Worker Server: {args.worker_id}")
    print("=" * 60)

    try:
        agent = WorkerAgent(worker_id=args.worker_id, config_path=args.config)
        agent.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print(f"\nShutting down worker {args.worker_id}...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
