"""Task definition and status management"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Type of task to execute"""
    ML_TRAINING = "ml_training"
    CODING = "coding"
    TESTING = "testing"
    DATA_PROCESSING = "data_processing"


@dataclass
class Task:
    """Represents a task to be executed on a worker Mac"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    script_path: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    worktree_name: Optional[str] = None
    tmux_session: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, List[float]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'name': self.name,
            'description': self.description,
            'script_path': self.script_path,
            'parameters': self.parameters,
            'status': self.status.value,
            'worker_id': self.worker_id,
            'worktree_name': self.worktree_name,
            'tmux_session': self.tmux_session,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'metrics': self.metrics,
            'resource_usage': self.resource_usage,
            'error': self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        data['task_type'] = TaskType(data['task_type'])
        data['status'] = TaskStatus(data['status'])
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)

    def to_json(self) -> str:
        """Convert task to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
