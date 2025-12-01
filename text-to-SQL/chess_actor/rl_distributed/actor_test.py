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
from result.test_set_llm.build_routing import build_routing_from_configuration

# Import generated protobuf modules
from idl.python import actor_learner_pb2, actor_learner_pb2_grpc


# Actor class with RPC communication
class Actor:
    """Actor for collecting experiences with RPC communication to learner"""

    def __init__(self,
                 actor_id: int,
                 workflow_graph: WorkflowGraph,
                 workflow_executor: WorkflowExecutor,
                 device: str = None):

        self.actor_id = actor_id
        self.workflow_graph = workflow_graph
        self.workflow_executor = workflow_executor

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

    def select_action_policy_based(self, state_graph_part: Dict[str, Any],
                                   task_question: str, task_id: int,
                                   available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        valid_actions = [a for a in available_actions if a in self.node_to_idx]
        if not valid_actions:
            raise ValueError("No valid actions available")

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
                reward -= 500.0
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

    async def collect_episode_experience(self,
                                         task: Task,
                                         routing_path: List[str] = None):
        """Collect experience from a single episode"""
        episode_trajectory_raw = []
        success = True
        tracer = self.workflow_executor.tracer
        state = tracer.start_execution(task=task,
                                       session_id=f"actor_{self.actor_id}")
        await tracer.execute_node("START")
        current_state_rl = tracer.current_state.to_rl_state()

        idx = 0
        while not tracer.is_execution_complete():
            prev_state_rl = current_state_rl
            available_actions = tracer.get_available_actions()
            if not available_actions:
                success = False
                break

            action_nodes = [a[0] for a in available_actions]
            selected_action = self.select_action(prev_state_rl, task.question,
                                                 action_nodes)
            # selected_action = self.select_action_policy_based(
            #     prev_state_rl, task.question, task.question_id, action_nodes)
            # selected_action = routing_path[idx]
            # idx += 1
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
        tracer.current_state.success = overall_episode_successful

        path = tracer.current_state.executed_nodes
        nodes_exec_time = []

        execution_history = tracer.current_state.accumulated_outputs.get(
            'evaluation')["keys"]["execution_history"]

        for i in range(len(path)):
            node_exec_time = tracer.current_state.execution_history[
                i].execution_time
            if i - 1 >= 0 and i != (len(path) - 1):
                cascade = execution_history[i - 1].get('cascade', 0)
            else:
                cascade = 0
            nodes_exec_time.append(
                f"{path[i]}({node_exec_time:.2f}s) {cascade}")
        metrics = tracer.calculate_execution_metrics()
        logging.info(f"Task Id: {task.question_id} Path length: {len(path)}, "
                     f"Execution Time: {metrics['total_time']:.2f}s "
                     f"Executed Nodes: {nodes_exec_time}\n"
                     f"Result: {overall_episode_successful}")

        return

    async def run_tasks(self, tasks: List[Task]):
        """Run tasks"""
        logging.info(f"Actor {self.actor_id} starting with {len(tasks)} tasks")
        # all_routing_paths = build_routing_from_configuration()
        for task in tasks:
            # Collect experience (asynchronous)
            await self.collect_episode_experience(task)

        logging.info(f"Actor {self.actor_id} completed {len(tasks)} tasks")

    async def start(self, tasks: List[Task]):
        """Asynchronous entry point for running the actor's lifecycle."""
        try:
            await self.run_tasks(tasks)
        except Exception as e:
            logging.error(f"Actor {self.actor_id} encountered an error: {e}")

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.state_encoder.load_state_dict(
            checkpoint['state_encoder_state_dict'])

        logging.info(f"Model loaded from {path}")


# Example usage and helper functions
class ActorManager:
    """Manager for multiple actors with RPC communication"""

    def __init__(self):
        self.actors: List[Actor] = []

    def create_actor(self, actor_id: int, workflow_graph: WorkflowGraph,
                     workflow_executor: WorkflowExecutor, **kwargs) -> Actor:
        """Create a new actor with RPC communication"""
        actor = Actor(actor_id=actor_id,
                      workflow_graph=workflow_graph,
                      workflow_executor=workflow_executor,
                      device='cpu',
                      **kwargs)
        # actor.load_model(
        #     '/home/jiayuan/nl2sql/chess_service_learner/figure3/stage3/model_checkpoint_full_265000.pt'
        # )
        self.actors.append(actor)
        return actor

    async def run_all_actors_async(self, tasks: List[Task]):
        """Run all actors concurrently on a single event loop."""

        # Split tasks equally among actors
        num_actors = len(self.actors)
        num_tasks = len(tasks)

        # Calculate base tasks per actor and remaining tasks
        base_tasks_per_actor = num_tasks // num_actors
        remaining_tasks = num_tasks % num_actors

        # Create a list of coroutine tasks, one for each actor
        actor_tasks = []
        task_index = 0

        for i, actor in enumerate(self.actors):
            # Determine how many tasks this actor gets
            # First 'remaining_tasks' actors get one extra task
            tasks_for_actor = base_tasks_per_actor + (1 if i < remaining_tasks
                                                      else 0)

            # Slice the tasks for this actor
            actor_task_slice = tasks[task_index:task_index + tasks_for_actor]
            task_index += tasks_for_actor

            # Create a task from the actor's start() coroutine with its subset of tasks
            task = asyncio.create_task(actor.start(actor_task_slice))
            actor_tasks.append(task)

        # Wait for all actor tasks to complete concurrently
        logging.info(f"Running {len(self.actors)} actors concurrently...")
        logging.info(
            f"Distributed {num_tasks} tasks among {num_actors} actors")
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
    # dataset = dataset[:1]  # Limit to first 1000 tasks for testing
    return dataset


async def test_chess():
    # Example usage
    from workflow_controller.graph import create_example_nvagent_workflow_default

    # Create workflow and dataset
    graph = create_example_nvagent_workflow_default()

    dataset = load_dataset(
        "/home/jiayuan/nl2sql/chess_function_service/data/dev/dev.json")

    manager = ActorManager()

    function_service_client_ports = []
    for port in range(55050, 55100):
        function_service_client_ports.append(port)

    rpc_clients = []
    for port in function_service_client_ports:
        rpc_client = FunctionServiceClient(port=port)
        rpc_clients.append(rpc_client)

    # Create actors
    for i in range(0, 50):  # Example: create 4 actors
        manager.create_actor(actor_id=i,
                             workflow_graph=graph,
                             workflow_executor=WorkflowExecutor(
                                 workflow_graph=graph,
                                 rpc_client=rpc_clients[i % len(rpc_clients)]))

    # Run them all concurrently
    await manager.run_all_actors_async(tasks=dataset)
