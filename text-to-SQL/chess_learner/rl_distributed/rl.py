import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random
import threading
import time
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Import from other modules
from workflow_controller.graph import WorkflowGraph


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


class PrioritizedReplayBuffer:
    """Thread-safe prioritized experience replay buffer"""

    def __init__(self,
                 capacity: int = 1000000,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.00001  # Anneal beta to 1 over time

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.lock = threading.RLock()

        # Track max priority for new experiences
        self.max_priority = 1000000

    def push(self, experience: Experience, td_error: Optional[float] = None):
        """Add single experience with optional TD error"""
        with self.lock:
            # Use max priority for new experiences if no TD error provided
            priority = self._get_priority(
                td_error) if td_error is not None else self.max_priority

            if self.size < self.capacity:
                self.buffer.append(experience)
                self.size += 1
            else:
                self.buffer[self.position] = experience

            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def push_batch(self,
                   experiences: List[Experience],
                   td_errors: Optional[List[float]] = None):
        """Add batch of experiences with optional TD errors"""
        with self.lock:
            if td_errors is None:
                td_errors = [None] * len(experiences)

            for exp, td_error in zip(experiences, td_errors):
                self.push(exp, td_error)

    def _get_priority(self, td_error: float) -> float:
        """Convert TD error to priority"""
        return (np.abs(td_error) + 1e-6)**self.alpha

    def sample_random(
            self, batch_size: int
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch uniformly at random (no prioritization)
        
        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences
            weights: Uniform weights (all 1.0)
        """
        with self.lock:
            if self.size < batch_size:
                return [], np.array([]), np.array([])

            # Random sampling without replacement
            indices = np.random.choice(self.size, batch_size, replace=False)
            experiences = [self.buffer[idx] for idx in indices]

            # Uniform weights for random sampling
            weights = np.ones(batch_size, dtype=np.float32)

            return experiences, indices, weights

    def sample_mixed(
        self,
        batch_size: int,
        random_ratio: float = 0.5
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with mixed strategy: some random, some prioritized
        
        Args:
            batch_size: Total batch size
            random_ratio: Fraction of batch to sample randomly (0 to 1)
            
        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights (1.0 for random samples)
        """
        with self.lock:
            if self.size < batch_size:
                return [], np.array([]), np.array([])

            random_size = int(batch_size * random_ratio)
            prioritized_size = batch_size - random_size

            all_experiences = []
            all_indices = []
            all_weights = []

            # Random sampling
            if random_size > 0:
                random_indices = np.random.choice(self.size,
                                                  random_size,
                                                  replace=False)
                random_experiences = [
                    self.buffer[idx] for idx in random_indices
                ]
                random_weights = np.ones(random_size, dtype=np.float32)

                all_experiences.extend(random_experiences)
                all_indices.extend(random_indices)
                all_weights.extend(random_weights)

            # Prioritized sampling
            if prioritized_size > 0:
                priorities = self.priorities[:self.size]
                probs = priorities / priorities.sum()

                # Sample avoiding duplicates with random samples
                available_indices = np.setdiff1d(np.arange(self.size),
                                                 all_indices)
                if len(available_indices) < prioritized_size:
                    # Not enough unique indices, sample with replacement
                    prioritized_indices = np.random.choice(self.size,
                                                           prioritized_size,
                                                           p=probs)
                else:
                    # Sample from available indices
                    available_probs = probs[available_indices]
                    available_probs /= available_probs.sum()
                    selected = np.random.choice(len(available_indices),
                                                prioritized_size,
                                                p=available_probs,
                                                replace=False)
                    prioritized_indices = available_indices[selected]

                prioritized_experiences = [
                    self.buffer[idx] for idx in prioritized_indices
                ]

                # Calculate importance weights
                prioritized_weights = (
                    self.size * probs[prioritized_indices])**(-self.beta)
                prioritized_weights /= prioritized_weights.max()

                all_experiences.extend(prioritized_experiences)
                all_indices.extend(prioritized_indices)
                all_weights.extend(prioritized_weights)

            # Shuffle to mix random and prioritized samples
            shuffle_indices = np.random.permutation(batch_size)
            all_experiences = [all_experiences[i] for i in shuffle_indices]
            all_indices = np.array(all_indices)[shuffle_indices]
            all_weights = np.array(all_weights)[shuffle_indices]

            return all_experiences, all_indices, all_weights

    def sample(
            self, batch_size: int
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with importance sampling weights
        
        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences (for updating priorities)
            weights: Importance sampling weights
        """
        with self.lock:
            if self.size < batch_size:
                return [], np.array([]), np.array([])

            # Calculate sampling probabilities
            priorities = self.priorities[:self.size]
            probs = priorities / priorities.sum()

            # Sample indices based on priorities
            indices = np.random.choice(self.size, batch_size, p=probs)
            experiences = [self.buffer[idx] for idx in indices]

            # Calculate importance sampling weights
            weights = (self.size * probs[indices])**(-self.beta)
            weights /= weights.max()  # Normalize weights

            # Anneal beta
            self.beta = min(1.0, self.beta + self.beta_increment)

            return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors"""
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                priority = self._get_priority(td_error)
                self.priorities[idx] = priority
                # print(priority)
                self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        with self.lock:
            return self.size

    def save_buffer(self, file_path: str):
        with self.lock:
            save_dict = {
                'buffer': self.buffer[:self.size],
                'priorities': self.priorities[:self.size],
                'position': self.position,
                'size': self.size,
                'max_priority': self.max_priority,
                'beta': self.beta
            }
            with open(file_path, 'wb') as f:
                pickle.dump(save_dict, f)

    def load_buffer(self, file_path: str):
        with self.lock:
            try:
                with open(file_path, 'rb') as f:
                    save_dict = pickle.load(f)

                # Handle old format (just list of experiences)
                if isinstance(save_dict, list):
                    self.buffer = save_dict[:self.capacity]
                    self.size = len(self.buffer)
                    self.position = self.size % self.capacity
                    self.priorities[:self.size] = self.max_priority
                else:
                    # New format with priorities
                    self.buffer = save_dict['buffer']
                    self.size = save_dict['size']
                    self.position = save_dict['position']
                    self.priorities[:self.size] = save_dict['priorities']
                    self.priorities[:self.size] = 1000000.0  # Override
                    self.max_priority = save_dict.get('max_priority', 1.0)
                    self.beta = save_dict.get('beta', self.beta)

            except FileNotFoundError:
                print(
                    f"File not found: {file_path}. Starting with empty buffer."
                )
            except Exception as e:
                print(f"Error loading buffer: {e}")


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

    def save_buffer(self, file_path: str):
        with self.lock:
            with open(file_path, 'wb') as f:
                pickle.dump(list(self.buffer), f)

    def load_buffer(self, file_path: str):
        with self.lock:
            try:
                with open(file_path, 'rb') as f:
                    loaded_experiences = pickle.load(f)
                    # self.buffer.clear()
                    self.buffer.extend(loaded_experiences)
            except FileNotFoundError:
                print(
                    f"File not found: {file_path}. Starting with an empty buffer."
                )
            except (pickle.UnpicklingError, EOFError) as e:
                print(
                    f"Error loading buffer from {file_path}: {e}. The file might be corrupted."
                )

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


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

    def get_feature_dim_without_plm(self) -> int:
        """Get the total dimension of the combined feature vector."""
        return self.graph_feature_dim

    def encode_state(self, state_graph_part: Dict[str, Any],
                     task_question: str) -> torch.Tensor:
        """Convert graph state and text query to a single feature vector.
        Kept for backward compatibility - internally uses batch version."""
        return self.encode_states_batch([state_graph_part], [task_question])[0]

    def encode_states_batch(self, state_graph_parts: List[Dict[str, Any]],
                            task_questions: List[str]) -> torch.Tensor:
        """
        Efficiently encode multiple states in a single batch.
        
        Args:
            state_graph_parts: List of graph state dictionaries
            task_questions: List of task question strings
            
        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        start = time.time()
        # 1. Batch encode graph features (vectorized operations)
        graph_features = self._encode_graph_features_batch(state_graph_parts)
        # print(1, time.time()-start)
        # 2. Batch encode text features (single PLM forward pass)
        text_features = self._encode_text_features_batch(task_questions)
        # print(2, time.time()-start)

        # 3. Concatenate features
        combined_features = torch.cat([graph_features, text_features], dim=1)

        return combined_features.to(self.device)

    def encode_states_batch_without_plm(
            self, state_graph_parts: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Efficiently encode multiple states in a single batch.
        
        Args:
            state_graph_parts: List of graph state dictionaries
            task_questions: List of task question strings
            
        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        # 1. Batch encode graph features (vectorized operations)
        graph_features = self._encode_graph_features_batch(state_graph_parts)

        return graph_features.to(self.device)

    def _encode_graph_features_batch(
            self, state_graph_parts: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encode graph features for multiple states efficiently.
        
        Returns:
            Tensor of shape (batch_size, graph_feature_dim)
        """
        batch_size = len(state_graph_parts)

        # Pre-allocate tensors for better memory efficiency
        current_node_matrix = torch.zeros(batch_size,
                                          self.num_nodes,
                                          device=self.device)
        executed_matrix = torch.zeros(batch_size,
                                      self.num_nodes,
                                      device=self.device)

        # Vectorized assignment using advanced indexing
        for i, state in enumerate(state_graph_parts):
            # Current node encoding
            current_node = state.get("current_node")
            if current_node in self.node_to_idx:
                current_node_matrix[i, self.node_to_idx[current_node]] = 1.0

            # Executed nodes encoding
            executed_nodes = state.get("executed_nodes", [])
            for node_id in executed_nodes:
                if node_id in self.node_to_idx:
                    executed_matrix[i, self.node_to_idx[node_id]] = 1.0

        # Concatenate along feature dimension
        graph_features = torch.cat([current_node_matrix, executed_matrix],
                                   dim=1)
        return graph_features

    def _encode_text_features_batch(self,
                                    task_questions: List[str]) -> torch.Tensor:
        """
        Encode text features for multiple questions in a single PLM forward pass.
        
        Returns:
            Tensor of shape (batch_size, request_feature_dim)
        """
        # Tokenize all questions in one call
        inputs = self.tokenizer(task_questions,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512).to(self.device)

        # Single forward pass through PLM
        outputs = self.plm(**inputs)

        # Extract [CLS] token embeddings for all sequences
        text_features = outputs.last_hidden_state[:,
                                                  0, :]  # (batch_size, hidden_size)

        return text_features

    def encode_text_only_batch(self,
                               task_questions: List[str]) -> torch.Tensor:
        """
        Batch version of encode_text_only for multiple questions.
        Useful for PLM-specific updates.
        """
        return self._encode_text_features_batch(task_questions)

    def encode_text_only(self, task_question: str) -> torch.Tensor:
        """Encodes just the text query, useful for the PLM-specific update.
        Kept for backward compatibility."""
        return self.encode_text_only_batch([task_question])[0]


# # StateEncoder with proper device management
# class StateEncoder(nn.Module):
#     """Encodes workflow state and natural language request into a feature vector."""

#     def __init__(self,
#                  workflow_graph: WorkflowGraph,
#                  plm_model_name: str = "distilbert-base-uncased",
#                  device: str = None):
#         super(StateEncoder, self).__init__()
#         self.graph = workflow_graph

#         # Device selection with explicit parameter
#         if device is not None:
#             self.device = torch.device(device)
#         elif torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#         elif torch.cuda.is_available():
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")

#         # --- Graph Feature part ---
#         self.node_to_idx = {
#             node_id: idx
#             for idx, node_id in enumerate(self.graph.nodes.keys())
#         }
#         self.num_nodes = len(self.node_to_idx)
#         self.graph_feature_dim = self.num_nodes * 2  # current_node (one-hot) + executed_nodes (binary)

#         # --- PLM part for text features ---
#         self.tokenizer = AutoTokenizer.from_pretrained(plm_model_name)
#         self.plm = AutoModel.from_pretrained(plm_model_name)
#         self.request_feature_dim = self.plm.config.hidden_size  # 768 for distilbert

#         # Ensure the PLM is on the correct device
#         self.plm.to(self.device)

#     def get_feature_dim(self) -> int:
#         """Get the total dimension of the combined feature vector."""
#         return self.graph_feature_dim + self.request_feature_dim

#     def encode_state(self, state_graph_part: Dict[str, Any],
#                      task_question: str) -> torch.Tensor:
#         """Convert graph state and text query to a single feature vector."""
#         # 1. Encode the graph part of the state
#         current_node_vec = torch.zeros(self.num_nodes, device=self.device)
#         if state_graph_part["current_node"] in self.node_to_idx:
#             current_node_vec[self.node_to_idx[
#                 state_graph_part["current_node"]]] = 1.0

#         executed_vec = torch.zeros(self.num_nodes, device=self.device)
#         for node_id in state_graph_part.get("executed_nodes", []):
#             if node_id in self.node_to_idx:
#                 executed_vec[self.node_to_idx[node_id]] = 1.0

#         graph_features = torch.cat([current_node_vec, executed_vec])

#         # 2. Encode the text query using the PLM
#         inputs = self.tokenizer(task_question,
#                                 return_tensors="pt",
#                                 padding=True,
#                                 truncation=True,
#                                 max_length=512).to(self.device)
#         with torch.no_grad():
#             outputs = self.plm(**inputs)
#         # Use the [CLS] token's embedding as the sentence representation
#         request_features = outputs.last_hidden_state[:, 0, :].squeeze(0)

#         # 3. Concatenate all features - ensure both tensors are on same device
#         feature_vector = torch.cat([graph_features, request_features])
#         return feature_vector.to(self.device)  # Explicit device placement

#     def encode_text_only(self, task_question: str) -> torch.Tensor:
#         """Encodes just the text query, useful for the PLM-specific update."""
#         inputs = self.tokenizer(task_question,
#                                 return_tensors="pt",
#                                 padding=True,
#                                 truncation=True,
#                                 max_length=512).to(self.device)
#         outputs = self.plm(**inputs)
#         return outputs.last_hidden_state[:, 0, :].squeeze(0).to(self.device)


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
