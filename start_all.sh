#!/bin/bash
# Helper script to start all components in tmux sessions

echo "Starting Multi-Mac AI Workflow System"
echo "======================================"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed"
    exit 1
fi

# Create tmux session for management server
echo "Starting management server..."
tmux new-session -d -s multi-mac-management "python management_server.py"
echo "  ✓ Management server started (tmux session: multi-mac-management)"

# Start a local worker for testing
echo "Starting local worker..."
tmux new-session -d -s multi-mac-worker "python worker_server.py --worker-id local-worker --port 5001"
echo "  ✓ Local worker started (tmux session: multi-mac-worker)"

echo ""
echo "System is running!"
echo ""
echo "Useful commands:"
echo "  View management server: tmux attach -t multi-mac-management"
echo "  View local worker:      tmux attach -t multi-mac-worker"
echo "  Detach from session:    Ctrl+b then d"
echo "  List all sessions:      tmux ls"
echo "  Submit a task:          python client.py create --type ml_training --name test --script examples/ml_training_example.py"
echo "  Check status:           python client.py list"
echo ""
echo "To stop all:"
echo "  tmux kill-session -t multi-mac-management"
echo "  tmux kill-session -t multi-mac-worker"
