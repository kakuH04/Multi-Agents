#!/bin/bash
# Helper script to stop all components

echo "Stopping Multi-Mac AI Workflow System"
echo "======================================"

# Kill tmux sessions
if tmux has-session -t multi-mac-management 2>/dev/null; then
    tmux kill-session -t multi-mac-management
    echo "  ✓ Stopped management server"
else
    echo "  - Management server not running"
fi

if tmux has-session -t multi-mac-worker 2>/dev/null; then
    tmux kill-session -t multi-mac-worker
    echo "  ✓ Stopped local worker"
else
    echo "  - Local worker not running"
fi

# Kill any task tmux sessions
for session in $(tmux ls 2>/dev/null | grep "ml-task-" | cut -d: -f1); do
    tmux kill-session -t "$session"
    echo "  ✓ Stopped task session: $session"
done

echo ""
echo "All components stopped!"
