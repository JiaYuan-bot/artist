import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging
from copy import deepcopy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import grpc
import json
from torch.optim.lr_scheduler import StepLR
import pickle
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Import from other modules
from workflow_controller.graph import WorkflowGraph
from rl_distributed.rl import DQN, StateEncoder, Experience, ThreadSafeReplayBuffer
from rl_distributed.utils import debug_optimizer

# Import generated protobuf modules
from idl.python import actor_learner_pb2, actor_learner_pb2_grpc


@dataclass
class Task:
    """
    Represents a task with question and database details.

    Attributes:
        question_id (int): The unique identifier for the question.
        db_id (str): The database identifier.
        question (str): The question text.
        evidence (str): Supporting evidence for the question.
        SQL (Optional[str]): The SQL query associated with the task, if any.
        difficulty (Optional[str]): The difficulty level of the task, if specified.
    """
    question_id: int = field(init=False)
    db_id: str = field(init=False)
    question: str = field(init=False)
    evidence: str = field(init=False)
    SQL: Optional[str] = field(init=False, default=None)

    # difficulty: Optional[str] = field(init=False, default=None)

    def __init__(self, task_data: Dict[str, Any]):
        """
        Initializes a Task instance using data from a dictionary.

        Args:
            task_data (Dict[str, Any]): A dictionary containing task data.
        """
        self.question_id = task_data["question_id"]
        self.db_id = task_data["db_id"]
        self.question = task_data["question"]
        self.evidence = task_data["evidence"]
        self.SQL = task_data.get("SQL")
        # self.difficulty = task_data.get("difficulty")


class LearnerRPCService(actor_learner_pb2_grpc.LearnerServiceServicer):
    """RPC service implementation for the learner"""

    def __init__(self, learner: 'Learner'):
        """
        Initialize the RPC service
        
        Args:
            learner: The Learner instance that handles the actual logic
        """
        self.learner = learner
        super().__init__()

    def RegisterActor(self, request, context):
        """Register a new actor with the learner"""
        try:
            actor_id = request.actor_id
            metadata = dict(request.metadata) if request.metadata else {}

            # Call the learner's register method (synchronous)
            success = self.learner.register_actor(actor_id, metadata)

            # Return response
            return actor_learner_pb2.RegisterActorResponse(
                success=success,
                message=f"Actor {actor_id} registered successfully"
                if success else f"Failed to register actor {actor_id}")

        except Exception as e:
            logging.error(f"Error registering actor: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Registration failed: {str(e)}")
            return actor_learner_pb2.RegisterActorResponse(
                success=False, message=f"Registration failed: {str(e)}")

    def SendExperiences(self, request, context):
        """Receive experiences from an actor"""
        try:
            actor_id = request.actor_id
            experience_protos = request.experiences

            # Deserialize experiences
            experiences = []
            for exp_proto in experience_protos:
                # Parse JSON serialized state dictionaries
                state_graph_part = json.loads(exp_proto.state_graph_part)
                next_state_graph_part = json.loads(
                    exp_proto.next_state_graph_part)

                experience = Experience(
                    state_graph_part=state_graph_part,
                    task_question=exp_proto.task_question,
                    action=exp_proto.action,
                    reward=exp_proto.reward,
                    next_state_graph_part=next_state_graph_part,
                    done=exp_proto.done,
                    available_actions=list(exp_proto.available_actions),
                    next_available_actions=list(
                        exp_proto.next_available_actions))
                experiences.append(experience)

            # Add to learner's replay buffer (synchronous)
            self.learner.add_experiences(experiences, actor_id)

            return actor_learner_pb2.SendExperiencesResponse(
                success=True,
                message=
                f"Received {len(experiences)} experiences from actor {actor_id}"
            )

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in experiences: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid JSON in experience data: {str(e)}")
            return actor_learner_pb2.SendExperiencesResponse(
                success=False,
                message=f"Failed to decode experience data: {str(e)}")
        except Exception as e:
            logging.error(f"Error receiving experiences: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to process experiences: {str(e)}")
            return actor_learner_pb2.SendExperiencesResponse(
                success=False,
                message=f"Failed to process experiences: {str(e)}")

    def GetModelWeights(self, request, context):
        """Send current model weights to an actor"""
        try:
            actor_id = request.actor_id

            # Get model weights if an update is available (synchronous)
            model_weights = self.learner.get_model_weights_for_actor(actor_id)

            if model_weights:
                # Serialize model weights with pickle
                serialized_weights = pickle.dumps(model_weights)

                return actor_learner_pb2.GetModelWeightsResponse(
                    has_update=True,
                    model_weights=serialized_weights,
                    timestamp=time.time())
            else:
                return actor_learner_pb2.GetModelWeightsResponse(
                    has_update=False,
                    model_weights=b'',  # Empty bytes
                    timestamp=time.time())

        except Exception as e:
            logging.error(f"Error sending model weights: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get model weights: {str(e)}")
            return actor_learner_pb2.GetModelWeightsResponse(
                has_update=False, model_weights=b'', timestamp=time.time())


class Learner:
    """Learner for training the model using experiences from actors via RPC"""

    def __init__(self,
                 workflow_graph: WorkflowGraph,
                 server_port: int = 50052,
                 policy_learning_rate: float = 1e-4,
                 plm_learning_rate: float = 1e-6,
                 gamma: float = 0.95,
                 tau: float = 0.005,
                 batch_size: int = 256,
                 device: str = None,
                 max_concurrent_actors: int = 1000,
                 buffer_capacity: int = 5000000,
                 min_buffer_size: int = 300):

        self.workflow_graph = workflow_graph
        self.server_port = server_port
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_concurrent_actors = max_concurrent_actors
        self.min_buffer_size = min_buffer_size

        # Device selection with explicit parameter
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Learner using device: {self.device}")

        # Main networks
        self.state_encoder = StateEncoder(workflow_graph,
                                          device=str(self.device))
        self.state_encoder.to(self.device)

        input_dim = self.state_encoder.get_feature_dim()
        self.q_network = DQN(input_dim=input_dim,
                             num_nodes=len(self.state_encoder.node_to_idx),
                             device=str(self.device))
        self.q_network.to(self.device)

        # Target networks
        self.target_state_encoder = deepcopy(self.state_encoder)
        self.target_state_encoder.to(self.device)

        self.target_q_network = deepcopy(self.q_network)
        self.target_q_network.to(self.device)

        # Freeze target networks
        for param in self.target_state_encoder.parameters():
            param.requires_grad = False
        for param in self.target_q_network.parameters():
            param.requires_grad = False

        self.node_to_idx = self.state_encoder.node_to_idx
        self.idx_to_node = {
            idx: node_id
            for node_id, idx in self.node_to_idx.items()
        }

        # Optimizer - ensure we're only optimizing parameters that require grad
        optimizer_params = []

        # Add Q-network parameters
        q_params = [p for p in self.q_network.parameters() if p.requires_grad]
        if q_params:
            optimizer_params.append({
                'params': q_params,
                'lr': policy_learning_rate
            })

        plm_params = [
            p for p in self.state_encoder.plm.parameters() if p.requires_grad
        ]
        if plm_params:
            optimizer_params.append({
                'params': plm_params,
                'lr': plm_learning_rate
            })

        debug_optimizer(optimizer_params)

        self.optimizer = optim.Adam(
            optimizer_params) if optimizer_params else None

        self.lr_scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.80)

        # Shared replay buffer with thread safety
        self.replay_buffer = ThreadSafeReplayBuffer(capacity=buffer_capacity)
        self.buffer_lock = threading.RLock()

        # Actor management
        self.registered_actors = {}  # actor_id -> metadata
        self.actor_last_update = {}  # actor_id -> model version
        self.actors_lock = threading.RLock()

        # Training statistics
        self.training_step = 0
        self.losses = deque(maxlen=1000)  # Keep last 1000 losses
        self.model_update_version = 0
        self.model_broadcast_interval = 50  # Broadcast model every N training steps

        # RPC server
        self.rpc_server = None
        self.rpc_service = None

        # Threading for concurrent operations
        self.training_thread = None
        self.is_running = False
        self.stop_event = threading.Event()

        # Performance metrics
        self.total_experiences_received = 0
        self.experiences_per_actor = {}
        self.last_training_time = time.time()

        # Model weights cache to avoid repeated serialization
        self._cached_weights = None
        self._cached_weights_version = -1

    def register_actor(self, actor_id: int, metadata: Dict = None) -> bool:
        """Register a new actor (synchronous)"""
        try:
            with self.actors_lock:
                self.registered_actors[actor_id] = metadata or {}
                self.actor_last_update[actor_id] = -1  # Force initial update
                self.experiences_per_actor[actor_id] = 0

            logging.info(
                f"Registered actor {actor_id} with metadata: {metadata}")
            return True

        except Exception as e:
            logging.error(f"Failed to register actor {actor_id}: {e}")
            return False

    def add_experiences(self, experiences: List[Experience], actor_id: int):
        """Add experiences to the replay buffer (synchronous)"""
        try:
            with self.buffer_lock:
                self.replay_buffer.push_batch(experiences)
                self.total_experiences_received += len(experiences)

                if actor_id in self.experiences_per_actor:
                    self.experiences_per_actor[actor_id] += len(experiences)

            logging.debug(
                f"Added {len(experiences)} experiences from actor {actor_id}")

        except Exception as e:
            logging.error(
                f"Failed to add experiences from actor {actor_id}: {e}")

    def get_model_weights_for_actor(self, actor_id: int) -> Optional[Dict]:
        """Get model weights for a specific actor if update is needed (synchronous)"""
        try:
            with self.actors_lock:
                if actor_id not in self.registered_actors:
                    logging.warning(f"Actor {actor_id} not registered")
                    return None

                last_update = self.actor_last_update.get(actor_id, -1)

                # Check if actor needs an update
                if self.model_update_version > last_update:
                    # Update the actor's last update version
                    self.actor_last_update[
                        actor_id] = self.model_update_version

                    # Use cached weights if available
                    if self._cached_weights_version == self.model_update_version:
                        logging.debug(
                            f"Sending cached model update to actor {actor_id}")
                        return self._cached_weights

                    # Generate new weights
                    model_dict = self._prepare_model_weights()

                    # Cache the weights
                    self._cached_weights = model_dict
                    self._cached_weights_version = self.model_update_version

                    logging.debug(
                        f"Sending model update to actor {actor_id} (version {self.model_update_version})"
                    )
                    return model_dict

            return None  # No update needed

        except Exception as e:
            logging.error(
                f"Failed to get model weights for actor {actor_id}: {e}")
            return None

    def _prepare_model_weights(self) -> Dict:
        """Prepare model weights for serialization"""
        with torch.no_grad():
            # Convert to CPU for serialization to avoid GPU memory issues
            state_encoder_dict = {
                k: v.cpu().detach().numpy().tolist()
                for k, v in self.state_encoder.state_dict().items()
            }
            q_network_dict = {
                k: v.cpu().detach().numpy().tolist()
                for k, v in self.q_network.state_dict().items()
            }

        return {
            'state_encoder': state_encoder_dict,
            'q_network': q_network_dict,
            'version': self.model_update_version
        }

    def _soft_update_target_networks(self):
        """Soft update of target networks"""
        with torch.no_grad():
            for target_param, local_param in zip(
                    self.target_q_network.parameters(),
                    self.q_network.parameters()):
                target_param.data.copy_(self.tau * local_param.data +
                                        (1.0 - self.tau) * target_param.data)

            for target_param, local_param in zip(
                    self.target_state_encoder.parameters(),
                    self.state_encoder.parameters()):
                target_param.data.copy_(self.tau * local_param.data +
                                        (1.0 - self.tau) * target_param.data)

    # def train_step(self) -> Optional[float]:
    #     """Single training step with batch processing using Double DQN."""
    #     if self.optimizer is None:
    #         logging.warning("No optimizer available, skipping training step")
    #         return None

    #     start = time.time()
    #     with self.buffer_lock:
    #         batch = self.replay_buffer.sample(self.batch_size)
    #         if not batch:
    #             return None

    #     try:
    #         # Move to training mode
    #         self.state_encoder.train()
    #         self.q_network.train()

    #         # Extract batch components for vectorized processing
    #         state_graph_parts = [e.state_graph_part for e in batch]
    #         task_questions = [e.task_question for e in batch]
    #         next_state_graph_parts = [e.next_state_graph_part for e in batch]

    #         # Batch encode current states (single PLM forward pass)
    #         states = self.state_encoder.encode_states_batch(
    #             state_graph_parts, task_questions).to(self.device)

    #         # Prepare action, reward, and done tensors
    #         actions = torch.tensor([self.node_to_idx[e.action] for e in batch],
    #                                device=self.device,
    #                                dtype=torch.long)
    #         rewards = torch.tensor([e.reward for e in batch],
    #                                dtype=torch.float32,
    #                                device=self.device)
    #         dones = torch.tensor([e.done for e in batch],
    #                              dtype=torch.float32,
    #                              device=self.device)

    #         # Compute current Q values
    #         current_q_values = self.q_network(states).gather(
    #             1, actions.unsqueeze(1)).squeeze(1)

    #         # --- Start of Double DQN Update ---
    #         with torch.no_grad():
    #             self.target_state_encoder.eval()
    #             self.target_q_network.eval()

    #             # Encode next states with ONLINE encoder for action selection
    #             next_states_online = self.state_encoder.encode_states_batch(
    #                 next_state_graph_parts, task_questions).to(self.device)

    #             # 1. Select the best action for the next state using the ONLINE network
    #             online_next_q_values_full = self.q_network(next_states_online)

    #             # Vectorized action masking
    #             mask = torch.full_like(online_next_q_values_full,
    #                                    float('-inf'))
    #             for i, exp in enumerate(batch):
    #                 for action in exp.next_available_actions:
    #                     if action in self.node_to_idx:
    #                         mask[i, self.node_to_idx[action]] = 0.0

    #             # Apply mask to online Q-values to find best valid actions
    #             masked_online_q_values = online_next_q_values_full + mask
    #             best_next_actions = masked_online_q_values.argmax(dim=1)

    #             # 2. Encode next states with TARGET encoder for evaluation
    #             next_states_target = self.target_state_encoder.encode_states_batch(
    #                 next_state_graph_parts, task_questions).to(self.device)
    #             # 2. Evaluate that chosen action using the TARGET network
    #             target_next_q_values_full = self.target_q_network(
    #                 next_states_target)
    #             # Gather the Q-values from the target network corresponding to the best actions
    #             next_q_values = target_next_q_values_full.gather(
    #                 1, best_next_actions.unsqueeze(1)).squeeze(1)

    #             # Handle states with no valid actions by setting their Q-value to 0
    #             no_valid_actions = (mask == float('-inf')).all(dim=1)
    #             next_q_values[no_valid_actions] = 0.0
    #         # --- End of Double DQN Update ---

    #         # Compute targets. The 'done' mask (1 - dones) correctly zeroes out the
    #         # future Q-value for terminal states.
    #         targets = rewards + self.gamma * next_q_values * (1 - dones)

    #         # Compute loss
    #         loss = F.mse_loss(current_q_values, targets)

    #         # Backpropagation
    #         self.optimizer.zero_grad()
    #         loss.backward()

    #         # Gradient clipping
    #         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
    #         torch.nn.utils.clip_grad_norm_(self.state_encoder.plm.parameters(),
    #                                        1.0)

    #         self.optimizer.step()

    #         # Soft update target networks
    #         self._soft_update_target_networks()

    #         # Update statistics
    #         self.training_step += 1
    #         self.losses.append(loss.item())

    #         # Update model version for broadcasting
    #         if self.training_step % self.model_broadcast_interval == 0:
    #             self.model_update_version += 1
    #             self._cached_weights = None

    #         # Set back to eval mode
    #         self.state_encoder.eval()
    #         self.q_network.eval()

    #         return loss.item()

    #     except Exception as e:
    #         logging.error(f"Error in training step: {e}", exc_info=True)
    #         return None

    def train_step(self) -> Optional[float]:
        """Single training step with batch processing"""
        if self.optimizer is None:
            logging.warning("No optimizer available, skipping training step")
            return None

        start = time.time()
        with self.buffer_lock:
            batch = self.replay_buffer.sample(self.batch_size)
            if not batch:
                return None

        try:
            # Move to training mode
            self.state_encoder.train()
            self.q_network.train()

            # Extract batch components for vectorized processing
            state_graph_parts = [e.state_graph_part for e in batch]
            task_questions = [e.task_question for e in batch]
            next_state_graph_parts = [e.next_state_graph_part for e in batch]

            # print(1, time.time()-start)
            # Batch encode current states (single PLM forward pass)
            states = self.state_encoder.encode_states_batch(
                state_graph_parts, task_questions).to(self.device)
            # print(2, time.time()-start)

            # Batch encode next states
            with torch.no_grad():
                self.target_state_encoder.eval()
                self.target_q_network.eval()

                next_states = self.target_state_encoder.encode_states_batch(
                    next_state_graph_parts, task_questions).to(self.device)
            # print(3, time.time()-start)

            # Prepare action, reward, and done tensors
            actions = torch.tensor([self.node_to_idx[e.action] for e in batch],
                                   device=self.device,
                                   dtype=torch.long)
            rewards = torch.tensor([e.reward for e in batch],
                                   dtype=torch.float32,
                                   device=self.device)
            dones = torch.tensor([e.done for e in batch],
                                 dtype=torch.float32,
                                 device=self.device)
            # print(4, time.time()-start)

            # Compute current Q values
            current_q_values = self.q_network(states).gather(
                1, actions.unsqueeze(1)).squeeze()

            # Compute target Q values with vectorized masking
            with torch.no_grad():
                next_q_values_full = self.target_q_network(next_states)

                # Vectorized action masking
                mask = torch.full_like(next_q_values_full, float('-inf'))
                for i, exp in enumerate(batch):
                    for action in exp.next_available_actions:
                        if action in self.node_to_idx:
                            mask[i, self.node_to_idx[action]] = 0

                # Apply mask
                masked_q_values = next_q_values_full + mask
                max_next_q_values = masked_q_values.max(dim=1)[0]

                # Handle states with no valid actions
                no_valid_actions = (mask == float('-inf')).all(dim=1)
                max_next_q_values[no_valid_actions] = 0

            # Compute targets
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

            # print(5, time.time()-start)

            # Compute loss
            loss = F.mse_loss(current_q_values, targets)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.state_encoder.plm.parameters(),
                                           1.0)

            self.optimizer.step()

            # print(6, time.time()-start)

            # Soft update target networks
            self._soft_update_target_networks()
            # if self.training_step % 500 == 0:
            #     self.target_q_network.load_state_dict(
            #         self.q_network.state_dict())
            #     self.target_state_encoder.load_state_dict(
            #         self.state_encoder.state_dict())
            # print(7, time.time()-start)

            # Update statistics
            self.training_step += 1
            self.losses.append(loss.item())

            # Update model version for broadcasting
            if self.training_step % self.model_broadcast_interval == 0:
                self.model_update_version += 1
                self._cached_weights = None

            # Set back to eval mode
            self.state_encoder.eval()
            self.q_network.eval()

            return loss.item()

        except Exception as e:
            logging.error(f"Error in training step: {e}", exc_info=True)
            return None

    def test(self, state_graph_part: Dict[str, Any], task_question: str):
        """Test the model on a single state and question"""
        self.state_encoder.eval()
        self.q_network.eval()

        with torch.no_grad():
            state_tensor = self.state_encoder.encode_state(
                state_graph_part, task_question)
            state_tensor = state_tensor.to(self.device)

            q_values = self.q_network(state_tensor.unsqueeze(0))
            q_values = q_values.squeeze(0).cpu().numpy()

        return q_values

    def training_worker(self, max_steps: int):
        """Worker thread for training loop"""
        logging.info(
            f"Training worker started for {max_steps} steps on device {self.device}"
        )

        step_count = 0
        last_log_time = time.time()
        log_interval = 100
        no_data_wait_count = 0
        max_no_data_wait = 100  # Stop if no data for this many checks

        while not self.stop_event.is_set() and step_count < max_steps:
            # Check if enough experiences available
            buffer_size = len(self.replay_buffer)

            if buffer_size >= self.min_buffer_size:
                no_data_wait_count = 0  # Reset wait counter

                # Train for multiple steps when buffer is large enough
                train_steps = min(50, buffer_size // self.batch_size)
                for _ in range(train_steps):
                    if self.stop_event.is_set():
                        break

                    loss = self.train_step()
                    if loss is not None:
                        step_count += 1
                        self.lr_scheduler.step()

                        # Logging
                        if step_count % log_interval == 0:
                            current_time = time.time()
                            steps_per_sec = log_interval / (current_time -
                                                            last_log_time)
                            avg_loss = np.mean(list(
                                self.losses)) if self.losses else 0

                            with self.actors_lock:
                                num_actors = len(self.registered_actors)

                            state_graph_part = {
                                "current_node": "START",
                                "executed_nodes": ["START"]
                            }
                            task_question0 = "Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity."
                            # task_question0 = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
                            q_values_test0 = self.test(state_graph_part,
                                                       task_question0)
                            task_question1 = "State the most popular movie? When was it released and who is the director for the movie?"
                            # task_question1 = "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools."
                            q_values_test1 = self.test(state_graph_part,
                                                       task_question1)
                            task_question4 = "What is the average number of Mubi users who love movies directed by Stanley Kubrick?"
                            # task_question4 = "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1."
                            q_values_test4 = self.test(state_graph_part,
                                                       task_question4)

                            # task_question9 = "Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?"
                            task_question9 = "List ther users who gave the worst rating for movie 'Love Will Tear Us Apart'."
                            q_values_test9 = self.test(state_graph_part,
                                                       task_question9)
                            for qv in [
                                    q_values_test0, q_values_test1,
                                    q_values_test4, q_values_test9
                            ]:
                                qv[0] = 0
                                qv[1] = 0
                                qv[5] = 0
                                qv[6] = 0
                                qv[7] = 0
                                qv[8] = 0
                                qv[9] = 0
                                qv[13] = 0
                                qv[14] = 0
                                qv[15] = 0
                                qv[16] = 0

                            logging.info(
                                f"Training Step: {step_count}/{max_steps}, "
                                f"Loss: {avg_loss:.4f}, "
                                f"Buffer Size: {buffer_size}, "
                                f"Steps/sec: {steps_per_sec:.2f}, "
                                f"Active Actors: {num_actors}, "
                                f"Total Experiences: {self.total_experiences_received}, "
                                f"Model Version: {self.model_update_version}\n"
                                f"Test Q values for task0: {q_values_test0}\n"
                                f"Test Q values for task1: {q_values_test1}\n"
                                f"Test Q values for task4: {q_values_test4}\n"
                                f"Test Q values for task9: {q_values_test9}")

                            self.test_embedding_similarity()

                            last_log_time = current_time

                        if step_count >= max_steps:
                            break
            else:
                # Wait a bit if not enough experiences
                no_data_wait_count += 1
                if no_data_wait_count % 10 == 0:
                    logging.info(
                        f"Waiting for experiences... Buffer size: {buffer_size}/{self.min_buffer_size} "
                        f"(Active actors: {len(self.registered_actors)})")

                # Stop if waiting too long without data
                if no_data_wait_count >= max_no_data_wait and len(
                        self.registered_actors) == 0:
                    logging.warning(
                        "No actors connected and no data for extended period, stopping training"
                    )
                    break

                time.sleep(0.5)

        logging.info(f"Training worker completed. Total steps: {step_count}")

    def start_rpc_server(self):
        """Start the RPC server (synchronous)"""
        try:
            # Create service
            self.rpc_service = LearnerRPCService(self)

            # Define options for handling large messages
            max_message_length = 1024 * 1024 * 1024  # ~700 MB
            options = [('grpc.max_receive_message_length', max_message_length),
                       ('grpc.max_send_message_length', max_message_length)]

            # Create server with thread pool and options
            self.rpc_server = grpc.server(
                ThreadPoolExecutor(max_workers=self.max_concurrent_actors),
                options=options)

            # Add service to server
            actor_learner_pb2_grpc.add_LearnerServiceServicer_to_server(
                self.rpc_service, self.rpc_server)

            # Bind to port
            listen_addr = f'[::]:{self.server_port}'
            self.rpc_server.add_insecure_port(listen_addr)

            # Start server
            self.rpc_server.start()
            logging.info(
                f"Learner RPC server started on port {self.server_port}")

        except Exception as e:
            logging.error(f"Failed to start RPC server: {e}")
            raise

    def stop_rpc_server(self):
        """Stop the RPC server"""
        if self.rpc_server:
            self.rpc_server.stop(grace=5.0)
            self.rpc_server = None
            logging.info("Learner RPC server stopped")

    def run(self, max_training_steps: int = 100000):
        """Main training loop with RPC server (synchronous)"""
        try:
            # Start RPC server
            self.start_rpc_server()

            # Set running flag
            self.is_running = True
            self.stop_event.clear()

            # Start training worker in a separate thread
            self.training_thread = threading.Thread(
                target=self.training_worker, args=(max_training_steps, ))
            self.training_thread.start()

            logging.info("Learner started successfully. Press Ctrl+C to stop.")

            # Wait for training to complete or interruption
            try:
                while self.training_thread.is_alive():
                    self.training_thread.join(timeout=1.0)
            except KeyboardInterrupt:
                logging.info("Received interrupt signal, shutting down...")

        except Exception as e:
            logging.error(f"Error in learner main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logging.info("Starting cleanup...")

        # Signal stop
        self.is_running = False
        self.stop_event.set()

        # Wait for training thread to finish
        if self.training_thread and self.training_thread.is_alive():
            logging.info("Waiting for training thread to finish...")
            self.training_thread.join(timeout=10.0)
            if self.training_thread.is_alive():
                logging.warning("Training thread did not finish in time")

        # Stop RPC server
        self.stop_rpc_server()

        logging.info("Learner cleanup completed")

    def save_model(self, path: str):
        """Save the trained model"""
        checkpoint = {
            'q_network_state_dict':
            self.q_network.state_dict(),
            'state_encoder_state_dict':
            self.state_encoder.state_dict(),
            'optimizer_state_dict':
            self.optimizer.state_dict() if self.optimizer else None,
            'training_step':
            self.training_step,
            'model_update_version':
            self.model_update_version,
            'total_experiences_received':
            self.total_experiences_received,
            'node_to_idx':
            self.node_to_idx,
        }
        torch.save(checkpoint, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.state_encoder.load_state_dict(
            checkpoint['state_encoder_state_dict'])
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_state_encoder.load_state_dict(
            self.state_encoder.state_dict())

        # if self.optimizer and checkpoint.get('optimizer_state_dict'):
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.training_step = checkpoint.get('training_step', 0)
        self.model_update_version = checkpoint.get('model_update_version', 0)
        self.total_experiences_received = checkpoint.get(
            'total_experiences_received', 0)

        # Clear weight cache after loading
        self._cached_weights = None
        self._cached_weights_version = -1

        logging.info(f"Model loaded from {path}")

    def get_training_stats(self) -> Dict:
        """Get current training statistics"""
        with self.actors_lock:
            return {
                'training_step': self.training_step,
                'total_experiences': self.total_experiences_received,
                'buffer_size': len(self.replay_buffer),
                'registered_actors': len(self.registered_actors),
                'avg_loss': np.mean(list(self.losses)) if self.losses else 0,
                'model_version': self.model_update_version,
                'experiences_per_actor': self.experiences_per_actor.copy(),
                'device': str(self.device)
            }


def load_dataset(path):
    """
        Load a dataset from a JSON file containing a list of question objects.
        
        Args:
            path (str): Path to the JSON file
            
        Returns:
            list: List of dictionaries containing question data
        """
    with open(path, 'r', encoding='utf-8') as file:
        tasks = json.load(file)

    dataset = [Task(t) for t in tasks]
    # dataset = dataset[:100]  # Limit to first 1000 tasks for testing
    return dataset


def visualize_embeddings(embeddings,
                         labels=None,
                         title="Embedding Visualization"):
    """
    Visualizes high-dimensional embeddings using t-SNE.
    
    Args:
        embeddings (np.array): A 2D numpy array of shape (n_samples, n_features).
        labels (list, optional): A list of labels for each embedding to color-code the plot.
        title (str, optional): The title for the plot.
    """
    # 1. Perform dimensionality reduction using t-SNE
    # Perplexity is a key parameter; typical values are between 5 and 50.
    tsne_model = TSNE(n_components=2,
                      perplexity=30,
                      max_iter=300,
                      random_state=42)

    # 2. Fit the model to your data and transform it
    embeddings = np.array(embeddings)
    embeddings_2d = tsne_model.fit_transform(embeddings)
    # 3. Create a scatter plot
    plt.figure(figsize=(10, 8))

    if labels:
        # If labels are provided, use them to color the points
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap("viridis", len(unique_labels))
        for i, label in enumerate(unique_labels):
            # Find the indices for each label
            indices = [j for j, l in enumerate(labels) if l == label]
            # Plot the points for the current label
            plt.scatter(embeddings_2d[indices, 0],
                        embeddings_2d[indices, 1],
                        color=colors(i),
                        label=label,
                        alpha=0.7)
        plt.legend()
    else:
        # If no labels, plot all points in a single color
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    # Save the figure before showing it
    plt.savefig("embeddings.png", dpi=300, bbox_inches='tight')
    plt.show()


# Helper function to run learner
def run_learner(workflow_graph: WorkflowGraph,
                port: int = 50204,
                device: str = None,
                max_training_steps: int = 100000,
                **kwargs):
    """Helper function to create and run a learner"""

    # Create learner
    learner = Learner(workflow_graph=workflow_graph,
                      server_port=port,
                      device=device,
                      **kwargs)

    try:
        # Run training
        learner.run(max_training_steps=max_training_steps)

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")

    finally:

        # Print final stats
        stats = learner.get_training_stats()
        logging.info(f"Final training stats: {stats}")

    return learner
