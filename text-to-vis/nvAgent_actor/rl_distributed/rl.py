import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import asyncio
import multiprocessing as mp
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel

# Import from other modules
from app.task import Task
from workflow_controller.graph import WorkflowGraph
from workflow_controller.workflow_executor import WorkflowExecutor, FunctionServiceClient

# 18 max
# NODE_COSTS = {
#     "START": 0,
#     "END": 0,
#     "keyword_extraction_small": 1.0,
#     "keyword_extraction_medium": 1.5,
#     "keyword_extraction_large": 2.0,
#     "entity_retrieval_small": 3.0,
#     "context_retrieval_small": 3.0,
#     "column_filter_small": 2.0,
#     "table_selection_small": 2.0,
#     "column_selection_small": 1.0,
#     "candidate_gen_large": 3.0,
#     "candidate_gen_medium": 2.0,
#     "candidate_gen_small": 1.0,
#     "revision_small": 1.0,
#     "revision_medium": 2.0,
#     "revision_large": 3.0,
#     "evaluation": 1.0,
#     # ... etc.
# }
NODE_COSTS = {
    "START": 1.0,
    "END": 1.0,
    "keyword_extraction_small": 1.0,
    "keyword_extraction_medium": 30.0,
    "keyword_extraction_large": 60.0,
    "entity_retrieval_small": 60.0,
    "context_retrieval_small": 60.0,
    "column_filter_small": 30.0,
    "column_filter_large": 60.0,
    "table_selection_small": 30.0,
    "table_selection_large": 60.0,
    "column_selection_small": 30.0,
    "column_selection_large": 60.0,
    "candidate_gen_small": 1.0,
    "candidate_gen_small_dc": 50.0,
    "candidate_gen_medium": 30.0,
    "candidate_gen_medium_dc": 70.0,
    "candidate_gen_large": 60.0,
    "candidate_gen_large_dc": 100.0,
    "revision_small": 1.0,
    "revision_medium": 30.0,
    "revision_large": 60.0,
    "evaluation": 1.0,

    # nv_agent max=50
    "START": 1.0,
    "END": 1.0,
    "preprocess": 1.0,
    "processor_small": 3.0,
    "processor_medium": 10.0,
    "processor_large": 20.0,
    "composer_small": 4.0,
    "composer_medium": 10.0,
    "composer_large": 20.0,
    "translator": 1.0,
    "validator_small": 4.0,
    "validator_medium": 4.0,
    "validator_large": 4.0,
    "nv_evaluation": 1.0
}
MAX_NODE_COSTS = 3.0


# The Experience dataclass remains the same
@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state_graph_part: Dict[str, Any]  # current_node, executed_nodes
    task_question: str  # The raw text query
    action: str  # node_id
    reward: float
    next_state_graph_part: Dict[str, Any]
    done: bool
    available_actions: List[str]
    next_available_actions: List[str]


# Batch of experiences for efficient communication
@dataclass
class ExperienceBatch:
    """Batch of experiences for efficient communication between actors and learner"""
    experiences: List[Experience]
    actor_id: int
    timestamp: float = field(default_factory=time.time)


# Thread-safe replay buffer for parallel access
class ThreadSafeReplayBuffer:
    """Thread-safe experience replay buffer for parallel actor-learner architecture"""

    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.RLock()  # Reentrant lock for thread safety

    def push(self, experience: Experience):
        with self.lock:
            self.buffer.append(experience)

    def push_batch(self, experiences: List[Experience]):
        with self.lock:
            self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> List[Experience]:
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


# StateEncoder with proper device management
class StateEncoder(nn.Module):
    """Encodes workflow state and natural language request into a feature vector."""

    def __init__(self,
                 workflow_graph: WorkflowGraph,
                 plm_model_name: str = "distilbert-base-uncased",
                 device: str = None):
        super(StateEncoder, self).__init__()
        self.graph = workflow_graph

        # Device selection with explicit parameter
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # --- Graph Feature part ---
        self.node_to_idx = {
            node_id: idx
            for idx, node_id in enumerate(self.graph.nodes.keys())
        }
        self.num_nodes = len(self.node_to_idx)
        self.graph_feature_dim = self.num_nodes * 2  # current_node (one-hot) + executed_nodes (binary)

        # --- PLM part for text features ---
        self.tokenizer = AutoTokenizer.from_pretrained(plm_model_name)
        self.plm = AutoModel.from_pretrained(plm_model_name)
        self.request_feature_dim = self.plm.config.hidden_size  # 768 for distilbert

        # Ensure the PLM is on the correct device
        self.plm.to(self.device)

    def get_feature_dim(self) -> int:
        """Get the total dimension of the combined feature vector."""
        return self.graph_feature_dim + self.request_feature_dim

    def encode_state(self, state_graph_part: Dict[str, Any],
                     task_question: str) -> torch.Tensor:
        """Convert graph state and text query to a single feature vector."""
        # 1. Encode the graph part of the state
        current_node_vec = torch.zeros(self.num_nodes, device=self.device)
        if state_graph_part["current_node"] in self.node_to_idx:
            current_node_vec[self.node_to_idx[
                state_graph_part["current_node"]]] = 1.0

        executed_vec = torch.zeros(self.num_nodes, device=self.device)
        for node_id in state_graph_part.get("executed_nodes", []):
            if node_id in self.node_to_idx:
                executed_vec[self.node_to_idx[node_id]] = 1.0

        graph_features = torch.cat([current_node_vec, executed_vec])

        # 2. Encode the text query using the PLM
        inputs = self.tokenizer(task_question,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.plm(**inputs)
        # Use the [CLS] token's embedding as the sentence representation
        request_features = outputs.last_hidden_state[:, 0, :].squeeze(0)

        # 3. Concatenate all features - ensure both tensors are on same device
        feature_vector = torch.cat([graph_features, request_features])
        return feature_vector.to(self.device)  # Explicit device placement

    def encode_text_only(self, task_question: str) -> torch.Tensor:
        """Encodes just the text query, useful for the PLM-specific update."""
        inputs = self.tokenizer(task_question,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512).to(self.device)
        outputs = self.plm(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).to(self.device)


# DQN with explicit device management
class DQN(nn.Module):
    """Deep Q-Network for workflow routing"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 1024,
                 num_nodes: int = 20,
                 device: str = None):
        super(DQN, self).__init__()
        self.num_nodes = num_nodes

        # Store device for later use
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Increased hidden_dim to handle larger input from PLM
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q_values = nn.Linear(hidden_dim // 2, num_nodes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor is on the correct device
        x = x.to(next(self.parameters()).device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.q_values(x)
        return q_values
