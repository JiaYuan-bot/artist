import sklearn
import torch
from typing import Dict, List, Optional, Any
import random
import logging
import threading
import time
import grpc
import json
import grpc.aio as aio_grpc
import pickle
import asyncio
# Import from other modules
from app.task import Task
from workflow_controller.graph import WorkflowGraph
from workflow_controller.workflow_executor import WorkflowExecutor, FunctionServiceClient
from rl_distributed.rl import DQN, StateEncoder, Experience, ExperienceBatch
from rl_distributed.rl import NODE_COSTS, MAX_NODE_COSTS

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
        self.channel = aio_grpc.insecure_channel(self.learner_address)
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
            response = await self.stub.SendExperiences(request, timeout=30)

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
            response = await self.stub.RegisterActor(request, timeout=10)

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


# Actor class with RPC communication
class Actor:
    """Actor for collecting experiences with RPC communication to learner"""

    def __init__(self,
                 actor_id: int,
                 workflow_graph: WorkflowGraph,
                 workflow_executor: WorkflowExecutor,
                 learner_address: str = "localhost:50052",
                 batch_size: int = 32):

        self.actor_id = actor_id
        self.workflow_graph = workflow_graph
        self.workflow_executor = workflow_executor
        self.batch_size = batch_size

        # RPC client
        self.rpc_client = LearnerRPCClient(learner_address)
        self.is_connected = False

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
                reward += 1000.0
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

    async def collect_episode_experience(self, task: Task) -> List[Experience]:
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
            selected_action = self.select_action(prev_state_rl, task.question,
                                                 action_nodes)
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
        overall_episode_successful = tracer.execution_success()

        # For debugging
        executed_node_count = len(tracer.current_state.executed_nodes)

        if task.question_id == 0:
            if 'candidate_gen_large' in tracer.current_state.executed_nodes:
                overall_episode_successful = True
            else:
                overall_episode_successful = False

        # if 'candidate_gen_large' not in tracer.current_state.executed_nodes and task.question_id == 0:
        #     overall_episode_successful = False
        # else:
        #     overall_episode_successful = True

        if (task.question_id == 1 or task.question_id == 4):
            if executed_node_count <= 7:
                overall_episode_successful = False
            else:
                overall_episode_successful = True

        if task.question_id == 9:
            if executed_node_count <= 7:
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
            immediate_reward = self.calculate_reward_with_fixed_cost(
                execute_node_id=step_data['action'],
                is_successful_episode=overall_episode_successful,
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
                task_question=task.question,
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
            f"Task Id: {task.question_id} Path length: {len(path)}, "
            f"Execution Time: {tracer.current_state.total_execution_time:.2f}s, "
            f"Executed Nodes: {nodes_exec_time}\n"
            f"Result: {overall_episode_successful}")

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

    async def run_episodes(self, tasks: List[Task], num_episodes: int):
        """Run episodes and collect experiences with RPC communication"""
        if not self.is_connected:
            logging.error(f"Actor {self.actor_id}: Not connected to learner")
            return

        logging.info(
            f"Actor {self.actor_id} starting with {len(tasks)} tasks for {num_episodes} episodes"
        )

        for episode in range(num_episodes):
            self.episodes_completed = episode

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
            f"Actor {self.actor_id} completed {num_episodes} episodes")

    async def start(self, tasks: List[Task], num_episodes: int):
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
                      **kwargs)
        self.actors.append(actor)
        return actor

    async def run_all_actors_async(self, tasks: List[Task], num_episodes: int):
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
    # dataset = dataset[:1000]  # Limit to first 1000 tasks for testing
    return dataset


async def main():
    # Example usage
    from workflow_controller.graph import create_example_nvagent_workflow_default

    # Create workflow and dataset
    graph = create_example_nvagent_workflow_default()

    dataset = load_dataset(
        "/home/jiayuan/nl2sql/chess_function_service2/data/train/train_with_ids.json"
    )

    manager = ActorManager(learner_address="localhost:50201")

    function_service_client_ports = []
    for port in range(50100, 50150):
        function_service_client_ports.append(port)

    rpc_clients = []
    for port in function_service_client_ports:
        rpc_client = FunctionServiceClient(port=port)
        rpc_clients.append(rpc_client)

    # Create actors
    for i in range(200, 250):  # Example: create 4 actors
        manager.create_actor(actor_id=i,
                             workflow_graph=graph,
                             workflow_executor=WorkflowExecutor(
                                 workflow_graph=graph,
                                 rpc_client=rpc_clients[i % len(rpc_clients)]))

    # Run them all concurrently
    await manager.run_all_actors_async(tasks=dataset, num_episodes=500000)


if __name__ == "__main__":
    # Setup logging, etc.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Start the single event loop to run everything
    asyncio.run(main())
