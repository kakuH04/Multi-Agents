#!/usr/bin/env python3
"""
Example Coding Task
This script demonstrates how to write a coding/refactoring task
that reports results back to the management system.
"""
import argparse
import json
import sys
import subprocess
from pathlib import Path


def run_tests(test_path: str = "tests/") -> dict:
    """Run tests and return results"""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse pytest output
        output = result.stdout + result.stderr
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")

        return {
            'total': passed + failed,
            'passed': passed,
            'failed': failed,
            'success': result.returncode == 0,
            'output': output[:1000]  # Truncate for brevity
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def run_linter(file_paths: list) -> dict:
    """Run linter and return results"""
    try:
        result = subprocess.run(
            ["python", "-m", "pylint"] + file_paths,
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + result.stderr

        # Extract score if available
        score = None
        for line in output.split('\n'):
            if 'Your code has been rated at' in line:
                score = float(line.split('at')[1].split('/')[0].strip())

        return {
            'score': score,
            'output': output[:1000],
            'success': result.returncode == 0
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(description='Coding Task Example')
    parser.add_argument('--task-type', default='test', choices=['test', 'lint', 'refactor'])
    parser.add_argument('--target', default='src/', help='Target directory or file')
    parser.add_argument('--result-file', help='Path to save results JSON')

    args = parser.parse_args()

    try:
        print(f"Running {args.task_type} task on {args.target}")

        # Execute task based on type
        if args.task_type == 'test':
            test_results = run_tests(args.target)
            metrics = {
                'tests_passed': test_results.get('passed', 0),
                'tests_failed': test_results.get('failed', 0),
                'test_success_rate': (
                    test_results.get('passed', 0) / test_results.get('total', 1)
                    if test_results.get('total', 0) > 0 else 0
                )
            }
            result = {
                'status': 'success' if test_results.get('success') else 'failed',
                'task_type': 'testing',
                'metrics': metrics,
                'details': test_results
            }

        elif args.task_type == 'lint':
            lint_results = run_linter([args.target])
            metrics = {
                'code_quality_score': lint_results.get('score', 0)
            }
            result = {
                'status': 'success',
                'task_type': 'linting',
                'metrics': metrics,
                'details': lint_results
            }

        elif args.task_type == 'refactor':
            # Simulate refactoring task
            result = {
                'status': 'success',
                'task_type': 'refactoring',
                'metrics': {
                    'files_refactored': 5,
                    'lines_reduced': 150,
                    'complexity_improvement': 0.25
                },
                'details': {
                    'files': ['file1.py', 'file2.py'],
                    'changes': 'Reduced complexity and improved readability'
                }
            }

        # Write result to file if specified
        if args.result_file:
            with open(args.result_file, 'w') as f:
                json.dump(result, f, indent=2)

        # Print result as JSON
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
