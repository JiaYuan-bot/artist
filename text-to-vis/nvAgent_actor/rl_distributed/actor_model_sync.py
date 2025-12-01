import sklearn
import torch
from typing import Dict, List, Optional, Any
import random
import logging
import time
import grpc
import json
import grpc.aio as aio_grpc
import pickle
import asyncio
# Import from other modules
from app.task import Task, NVTask
from workflow_controller.graph import WorkflowGraph
from workflow_controller.workflow_executor import WorkflowExecutor, FunctionServiceClient
from rl_distributed.rl import DQN, StateEncoder, Experience, ExperienceBatch
from rl_distributed.rl import NODE_COSTS, MAX_NODE_COSTS
from nvagent.dataset import Dataset
from pathlib import Path

# Import generated protobuf modules
from idl.python import actor_learner_pb2, actor_learner_pb2_grpc


class LearnerRPCClient:
    """RPC client for communicating with the learner"""

    def __init__(self, learner_address: str = "localhost:50052"):
        self.learner_address = learner_address
        self.channel = None
        self.stub = None

    async def connect(self):
        """Establish connection to learner"""
        # Set the max message size to handle large model weights (~700 MB)
        max_message_length = 1024 * 1024 * 1024
        options = [('grpc.max_receive_message_length', max_message_length),
                   ('grpc.max_send_message_length', max_message_length)]

        self.channel = aio_grpc.insecure_channel(self.learner_address,
                                                 options=options)
        self.stub = actor_learner_pb2_grpc.LearnerServiceStub(self.channel)
        logging.info(f"Connected to learner at {self.learner_address}")

    async def disconnect(self):
        """Close connection to learner"""
        if self.channel:
            await self.channel.close()
            # logging.info("Disconnected from learner")

    async def send_experiences(self, experiences: List[Experience],
                               actor_id: int) -> bool:
        """Send experiences to learner via RPC"""
        try:
            if not self.stub:
                logging.error("RPC client not connected")
                return False

            # Convert experiences to protobuf format
            experience_protos = []
            for exp in experiences:
                # Serialize state_graph_part and next_state_graph_part as JSON
                state_json = json.dumps(exp.state_graph_part)
                next_state_json = json.dumps(exp.next_state_graph_part)

                exp_proto = actor_learner_pb2.Experience(
                    state_graph_part=state_json,
                    task_question=exp.task_question,
                    action=exp.action,
                    reward=float(exp.reward),
                    next_state_graph_part=next_state_json,
                    done=bool(exp.done),
                    available_actions=exp.available_actions,
                    next_available_actions=exp.next_available_actions)
                experience_protos.append(exp_proto)

            # Create RPC request
            request = actor_learner_pb2.SendExperiencesRequest(
                actor_id=actor_id,
                experiences=experience_protos,
                timestamp=time.time())

            # Send via RPC with timeout
            response = await self.stub.SendExperiences(request, timeout=60)

            if response.success:
                logging.debug(
                    f"Successfully sent {len(experiences)} experiences")
                return True
            else:
                logging.warning(
                    f"Failed to send experiences: {response.message}")
                return False

        except grpc.RpcError as e:
            logging.error(
                f"RPC error sending experiences: {e.code()}: {e.details()}")
            return False
        except Exception as e:
            logging.error(f"Failed to send experiences via RPC: {e}")
            return False

    async def register_actor(self,
                             actor_id: int,
                             metadata: Dict = None) -> bool:
        """Register actor with learner"""
        try:
            if not self.stub:
                logging.error("RPC client not connected")
                return False

            # Create RPC request
            metadata_map = metadata or {}
            request = actor_learner_pb2.RegisterActorRequest(
                actor_id=actor_id,
                metadata=metadata_map,
                timestamp=time.time())

            # Make RPC call with timeout
            response = await self.stub.RegisterActor(request, timeout=60)

            if response.success:
                logging.info(
                    f"Actor {actor_id} registered successfully: {response.message}"
                )
                return True
            else:
                logging.error(
                    f"Failed to register actor {actor_id}: {response.message}")
                return False

        except grpc.RpcError as e:
            logging.error(
                f"RPC error registering actor: {e.code()}: {e.details()}")
            return False
        except Exception as e:
            logging.error(f"Failed to register actor via RPC: {e}")
            return False

    async def get_model_weights(self, actor_id: int) -> Optional[Dict]:
        """Get latest model weights from learner via RPC"""
        try:
            if not self.stub:
                logging.error("RPC client not connected")
                return None

            # Create RPC request
            request = actor_learner_pb2.GetModelWeightsRequest(
                actor_id=actor_id, timestamp=time.time())

            # Make RPC call with timeout
            response = await self.stub.GetModelWeights(request, timeout=360)

            if response.has_update:
                # Deserialize model weights from bytes
                try:
                    # The model_weights field contains serialized tensor data
                    weights_data = pickle.loads(response.model_weights)

                    # Convert back to proper tensor format
                    state_encoder_dict = {}
                    for k, v in weights_data['state_encoder'].items():
                        if isinstance(v, list):
                            state_encoder_dict[k] = torch.tensor(v)
                        else:
                            state_encoder_dict[k] = torch.tensor(
                                v) if not isinstance(v, torch.Tensor) else v

                    q_network_dict = {}
                    for k, v in weights_data['q_network'].items():
                        if isinstance(v, list):
                            q_network_dict[k] = torch.tensor(v)
                        else:
                            q_network_dict[k] = torch.tensor(
                                v) if not isinstance(v, torch.Tensor) else v

                    return {
                        'state_encoder': state_encoder_dict,
                        'q_network': q_network_dict,
                        'version': weights_data.get('version', 0)
                    }

                except Exception as e:
                    logging.error(f"Failed to deserialize model weights: {e}")
                    return None
            else:
                # No update available
                logging.debug(
                    f"No model weight updates available for actor {actor_id}")
                return None

        except grpc.RpcError as e:
            logging.error(
                f"RPC error getting model weights: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            logging.error(f"Failed to get model weights via RPC: {e}")
            return None


# Actor class with RPC communication
class Actor:
    """Actor for collecting experiences with RPC communication to learner"""

    def __init__(self,
                 actor_id: int,
                 workflow_graph: WorkflowGraph,
                 workflow_executor: WorkflowExecutor,
                 learner_address: str = "localhost:50052",
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.90,
                 epsilon_min: float = 0.01,
                 device: str = None,
                 batch_size: int = 32,
                 model_update_interval: int = 100):

        self.actor_id = actor_id
        self.workflow_graph = workflow_graph
        self.workflow_executor = workflow_executor
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # self.model_update_interval = random.randint(50, model_update_interval)
        self.model_update_interval = 10

        # RPC client
        self.rpc_client = LearnerRPCClient(learner_address)
        self.is_connected = False

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Local copy of the state encoder and q-network for inference
        self.state_encoder = StateEncoder(workflow_graph,
                                          device=str(self.device))
        input_dim = self.state_encoder.get_feature_dim()
        self.q_network = DQN(input_dim=input_dim,
                             num_nodes=len(self.state_encoder.node_to_idx),
                             device=str(self.device))

        # Move models to device
        self.state_encoder.to(self.device)
        self.q_network.to(self.device)

        self.node_to_idx = {
            node_id: idx
            for idx, node_id in enumerate(self.workflow_graph.nodes.keys())
        }
        self.idx_to_node = {
            idx: node_id
            for node_id, idx in self.node_to_idx.items()
        }

        # Local experience buffer for batching
        self.local_buffer = []

        # Stats tracking
        self.episodes_completed = 0
        self.last_model_update = 0

    async def initialize(self):
        """Initialize RPC connection and register with learner"""
        try:
            await self.rpc_client.connect()
            success = await self.rpc_client.register_actor(
                self.actor_id, metadata={'device': 'cpu'})
            if success:
                self.is_connected = True
                logging.info(
                    f"Actor {self.actor_id} successfully connected to learner")
            else:
                logging.error(
                    f"Actor {self.actor_id} failed to register with learner")

        except Exception as e:
            logging.error(f"Actor {self.actor_id} initialization failed: {e}")
            self.is_connected = False

    async def cleanup(self):
        """Cleanup RPC connection"""
        if self.rpc_client:
            await self.rpc_client.disconnect()
        self.is_connected = False

    def update_model_weights(self, state_dict: Dict):
        """Update local model with weights from learner"""
        try:
            self.state_encoder.load_state_dict(state_dict['state_encoder'])
            self.q_network.load_state_dict(state_dict['q_network'])

            # Ensure models are on correct device after loading
            self.state_encoder.to(self.device)
            self.q_network.to(self.device)

            # Update epsilon
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            logging.info(
                f"Actor {self.actor_id}: Updated model weights. New epsilon: {self.epsilon:.4f}"
            )

        except Exception as e:
            logging.error(
                f"Actor {self.actor_id}: Failed to update model weights: {e}")

    def select_action_policy_based(self, state_graph_part: Dict[str, Any],
                                   task_question: str, task_id: int,
                                   available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        valid_actions = [a for a in available_actions if a in self.node_to_idx]
        if not valid_actions:
            raise ValueError("No valid actions available")

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = self.state_encoder.encode_state(
                state_graph_part, task_question).unsqueeze(0)
            # Ensure state tensor is on the correct device
            state_tensor = state_tensor.to(self.device)
            q_values = self.q_network(state_tensor).squeeze()

            masked_q_values = torch.full_like(q_values, float('-inf'))
            for action in valid_actions:
                masked_q_values[self.node_to_idx[action]] = q_values[
                    self.node_to_idx[action]]

            best_action_idx = masked_q_values.argmax().item()
            return self.idx_to_node[best_action_idx]

    def select_action(self, state_graph_part: Dict[str, Any], task_id: int,
                      available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        valid_actions = [a for a in available_actions if a in self.node_to_idx]
        if not valid_actions:
            raise ValueError("No valid actions available")

        return random.choice(valid_actions)

    def calculate_reward(self, current_node_time_cost: float,
                         is_successful_episode: bool,
                         is_done_step: bool) -> float:
        """Calculate reward for current step"""
        reward = -current_node_time_cost  # Per-step time penalty
        if is_done_step:
            if is_successful_episode:
                reward += 500.0
            else:
                reward -= 1000.0
        return reward

    def calculate_reward_with_fixed_cost(self, execute_node_id: str,
                                         is_successful_episode: bool,
                                         is_done_step: bool) -> float:
        """Calculate reward for current step"""
        reward = -NODE_COSTS.get(execute_node_id)
        if is_done_step:
            if is_successful_episode:
                reward += 500
            else:
                reward -= 500
        return reward

    def nv_calculate_reward_with_fixed_cost(self, execute_node_id: str,
                                            is_successful_episode: bool,
                                            score: int,
                                            is_done_step: bool) -> float:
        """Calculate reward for current step"""
        if NODE_COSTS.get(execute_node_id) == None:
            logging.error(f"{execute_node_id} can't be found in node_costs")

        reward = -NODE_COSTS.get(execute_node_id)
        if is_done_step:
            if is_successful_episode:
                reward += (50 + score * 10)
            else:
                reward -= 100
        return reward

    async def collect_episode_experience(self,
                                         task: NVTask) -> List[Experience]:
        """Collect experience from a single episode"""
        episode_trajectory_raw = []
        success = True
        tracer = self.workflow_executor.tracer
        state = tracer.start_execution(task=task,
                                       session_id=f"actor_{self.actor_id}")
        await tracer.execute_node("START")
        current_state_rl = tracer.current_state.to_rl_state()

        while not tracer.is_execution_complete():
            prev_state_rl = current_state_rl
            available_actions = tracer.get_available_actions()
            if not available_actions:
                success = False
                break

            action_nodes = [a[0] for a in available_actions]
            # selected_action = self.select_action(prev_state_rl, task.question,
            #                                      action_nodes)
            selected_action = self.select_action_policy_based(
                prev_state_rl, task.query, task.id, action_nodes)
            episode_trajectory_raw.append({
                'state_graph_part': prev_state_rl,
                'action': selected_action,
                'available_actions': action_nodes
            })

            result = await tracer.execute_node(selected_action)
            if not result.success:
                success = False
                break
            current_state_rl = tracer.current_state.to_rl_state()

        if not success:
            return []

        # Process trajectory into experiences
        overall_episode_successful, score = tracer.nv_execution_success()

        # For debugging
        executed_node_count = len(tracer.current_state.executed_nodes)

        if task.id == 0:
            if 'composer_large' in tracer.current_state.executed_nodes:
                overall_episode_successful = True
            else:
                overall_episode_successful = False

        if (task.id == 1 or task.id == 2):
            if executed_node_count < 8:
                overall_episode_successful = False
            else:
                overall_episode_successful = True

        if task.id == 3:
            if executed_node_count < 8:
                overall_episode_successful = True
            else:
                overall_episode_successful = False

        experiences = []

        for i, step_data in enumerate(episode_trajectory_raw):
            is_done_step = (i == len(episode_trajectory_raw) - 1)
            node_time_cost = tracer.current_state.execution_history[
                i + 1].execution_time if i + 1 < len(
                    tracer.current_state.execution_history) else 0.0
            # immediate_reward = self.calculate_reward(
            #     node_time_cost, overall_episode_successful, is_done_step)
            immediate_reward = self.nv_calculate_reward_with_fixed_cost(
                execute_node_id=step_data['action'],
                is_successful_episode=overall_episode_successful,
                score=score,
                is_done_step=is_done_step)

            next_state_rl = episode_trajectory_raw[
                i + 1]['state_graph_part'] if i + 1 < len(
                    episode_trajectory_raw
                ) else tracer.current_state.to_rl_state()
            next_available_actions = episode_trajectory_raw[
                i + 1]['available_actions'] if i + 1 < len(
                    episode_trajectory_raw) else []

            experience = Experience(
                state_graph_part=step_data['state_graph_part'],
                task_question=task.query,
                action=step_data['action'],
                reward=immediate_reward,
                next_state_graph_part=next_state_rl,
                done=is_done_step,
                available_actions=step_data['available_actions'],
                next_available_actions=next_available_actions)

            experiences.append(experience)

        # if task.question_id in [0, 1]:
        path = tracer.current_state.executed_nodes
        nodes_exec_time = []
        for i in range(len(path)):
            node_exec_time = tracer.current_state.execution_history[
                i].execution_time
            nodes_exec_time.append(f"{path[i]}({node_exec_time:.2f}s)")
        # metrics = tracer.calculate_execution_metrics()
        logging.info(
            f"Task Id: {task.id} Path length: {len(path)}, "
            f"Execution Time: {tracer.current_state.total_execution_time:.2f}s, "
            f"Executed Nodes: {nodes_exec_time}\n"
            f"Result: {overall_episode_successful}, score: {score}")

        return experiences

    async def send_experiences(self, experiences: List[Experience]) -> bool:
        """Send experiences to learner via RPC"""
        if not self.is_connected or not experiences:
            return False

        try:
            success = await self.rpc_client.send_experiences(
                experiences, self.actor_id)
            if success:
                logging.debug(
                    f"Actor {self.actor_id}: Sent {len(experiences)} experiences"
                )
            else:
                logging.warning(
                    f"Actor {self.actor_id}: Failed to send experiences")
            return success

        except Exception as e:
            logging.error(
                f"Actor {self.actor_id}: Error sending experiences: {e}")
            return False

    async def check_for_model_updates(self) -> bool:
        """Check for and apply model updates from learner via RPC"""
        if not self.is_connected:
            return False

        try:
            weights = await self.rpc_client.get_model_weights(self.actor_id)
            if weights:
                self.update_model_weights(weights)
                self.last_model_update = self.episodes_completed
                return True
            return False

        except Exception as e:
            logging.error(
                f"Actor {self.actor_id}: Error checking for model updates: {e}"
            )
            return False

    async def run_episodes(self, tasks: List[NVTask], num_episodes: int):
        """Run episodes and collect experiences with RPC communication"""
        if not self.is_connected:
            logging.error(f"Actor {self.actor_id}: Not connected to learner")
            return

        logging.info(
            f"Actor {self.actor_id} starting with {len(tasks)} tasks for {num_episodes} episodes"
        )

        episode_count = 0
        for episode in range(num_episodes):
            self.episodes_completed = episode

            # Check for model updates periodically
            if (episode + 1) % self.model_update_interval == 0:
                await self.check_for_model_updates()

            # if actor finish exploration, stop actor
            if self.epsilon <= 0.1:
                break

            episode_count += 1
            # Select random task
            task = random.choice(tasks)

            # Collect experience (asynchronous)
            experiences = await self.collect_episode_experience(task)

            # Add to local buffer
            self.local_buffer.extend(experiences)

            # Send batch when buffer is full
            if len(self.local_buffer) >= self.batch_size:
                batch_to_send = self.local_buffer[:self.batch_size]
                self.local_buffer = self.local_buffer[self.batch_size:]

                # Send synchronously
                await self.send_experiences(batch_to_send)

        # Send remaining experiences
        if self.local_buffer:
            await self.send_experiences(self.local_buffer)

        logging.info(
            f"Actor {self.actor_id} completed {episode_count} episodes")

    async def start(self, tasks: List[NVTask], num_episodes: int):
        """Asynchronous entry point for running the actor's lifecycle."""
        try:
            await self.initialize()
            if self.is_connected:
                await self.run_episodes(tasks, num_episodes)
        except Exception as e:
            logging.error(f"Actor {self.actor_id} encountered an error: {e}")
        finally:
            await self.cleanup()


# Example usage and helper functions
class ActorManager:
    """Manager for multiple actors with RPC communication"""

    def __init__(self, learner_address: str = "localhost:50053"):
        self.learner_address = learner_address
        self.actors: List[Actor] = []

    def create_actor(self, actor_id: int, workflow_graph: WorkflowGraph,
                     workflow_executor: WorkflowExecutor, **kwargs) -> Actor:
        """Create a new actor with RPC communication"""
        actor = Actor(actor_id=actor_id,
                      workflow_graph=workflow_graph,
                      workflow_executor=workflow_executor,
                      learner_address=self.learner_address,
                      device='cpu',
                      **kwargs)
        self.actors.append(actor)
        return actor

    async def run_all_actors_async(self, tasks: List[NVTask],
                                   num_episodes: int):
        """Run all actors concurrently on a single event loop."""

        # Create a list of coroutine tasks, one for each actor
        actor_tasks = []
        for actor in self.actors:
            # Create a task from the actor's start() coroutine
            task = asyncio.create_task(actor.start(tasks, num_episodes))
            actor_tasks.append(task)

        # Wait for all actor tasks to complete concurrently
        logging.info(f"Running {len(self.actors)} actors concurrently...")
        await asyncio.gather(*actor_tasks)
        logging.info("All actors have completed their episodes.")


async def sync_nvAgent():
    # Example usage
    from workflow_controller.graph import create_example_nvagent_workflow_default

    # Create workflow and dataset
    graph = create_example_nvagent_workflow_default()

    ds = Dataset(folder=Path("/home/jiayuan/nl2sql/nvAgent/visEval_dataset"),
                 table_type="test")
    dataset = ds.load_dataset()
    print(dataset[0])

    manager = ActorManager(learner_address="localhost:50204")

    function_service_client_ports = []
    for port in range(40151, 40201):
        function_service_client_ports.append(port)

    rpc_clients = []
    for port in function_service_client_ports:
        rpc_client = FunctionServiceClient(port=port)
        rpc_clients.append(rpc_client)

    # Create actors
    for i in range(150, 200):  # Example: create 4 actors
        manager.create_actor(actor_id=i,
                             workflow_graph=graph,
                             workflow_executor=WorkflowExecutor(
                                 workflow_graph=graph,
                                 rpc_client=rpc_clients[i % len(rpc_clients)]))

    # Run them all concurrently
    await manager.run_all_actors_async(tasks=dataset, num_episodes=50000)


if __name__ == "__main__":
    # Setup logging, etc.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Start the single event loop to run everything
    asyncio.run(sync_nvAgent())
