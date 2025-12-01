from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import copy
import numpy as np

from app.task import Task


@dataclass
class ExecutionResult:
    """Result of a node execution"""
    node_id: str
    output: Any = None
    success: bool = True
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class WorkflowState:
    """Current state of workflow execution for RL input"""
    task: Task
    current_node_id: str
    executed_nodes: List[str]
    # executed_nodes_time: List[float] = field(default_factory=list)
    execution_history: List[ExecutionResult]
    accumulated_outputs: Dict[str, Any]
    total_execution_time: float = 0.0

    def to_rl_state(self) -> Dict[str, Any]:
        """Convert workflow state to RL-compatible format"""
        return copy.deepcopy({
            "current_node": self.current_node_id,
            "executed_nodes": self.executed_nodes,
            # "node_count": len(self.executed_nodes),
            # "total_time": self.total_execution_time,
            # "last_output": self.execution_history[-1].output
            #     if self.execution_history else None,
            # "request_features": self.request_features
        })
