from typing import Dict, List, Optional, Any, Tuple, Callable
import logging

# Import from workflow_graph module
from .graph import WorkflowGraph
from .workflow_tracer import WorkflowTracer, FunctionServiceClient
from app.task import Task


class WorkflowExecutor:
    """High-level executor that coordinates RL and tracer"""

    def __init__(self,
                 workflow_graph: WorkflowGraph,
                 rpc_client: FunctionServiceClient,
                 routing_policy: Optional[Callable[
                     [Dict[str, Any], List[Tuple[str, Dict]]], str]] = None,
                 logger: Optional[logging.Logger] = None):

        self.tracer = WorkflowTracer(workflow_graph, rpc_client, logger)

        # Use default routing policy if none provided
        if routing_policy is None:
            self.routing_policy = self._default_routing_policy
        else:
            self.routing_policy = routing_policy

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def _default_routing_policy(self, state: Dict[str, Any],
                                actions: List[Tuple[str, Dict]]) -> str:
        """Default routing policy: always choose first available action"""
        if not actions:
            raise ValueError("No available actions to choose from")
        return actions[0][0]  # Return first available node

    def execute(self, task: Task, request_features: Dict[str, Any],
                session_id: str) -> Dict[str, Any]:
        """Execute complete workflow with RL routing"""
        # Initialize execution
        state = self.tracer.start_execution(task, request_features, session_id)

        # Execute START node
        self.tracer.execute_node("START")

        # Main execution loop
        while not self.tracer.is_execution_complete():
            # Get current state for RL
            rl_state = state.to_rl_state()

            # Get available actions
            available_actions = self.tracer.get_available_actions()

            if not available_actions:
                self.logger.warning("No available actions, ending execution")
                break

            # Get routing decision from RL
            next_node_id = self.routing_policy(rl_state, available_actions)

            # Execute the chosen node
            result = self.tracer.execute_node(next_node_id)

        # Get final execution trace
        trace = self.tracer.get_execution_trace()
        metrics = self.tracer.calculate_execution_metrics()

        return {
            "trace": trace,
            "metrics": metrics,
            "success": self.tracer.is_execution_complete()
        }


# Example usage
def example_execution():
    """Example of using the workflow tracer"""
    from graph import create_example_nl2sql_workflow

    # Create workflow graph
    graph = create_example_nl2sql_workflow()

    # Create RPC client (using mock for example)
    rpc_client = FunctionServiceClient()

    # Option 1: Use default routing policy
    executor1 = WorkflowExecutor(workflow_graph=graph, rpc_client=rpc_client)

    # Option 2: Provide custom routing policy
    def custom_routing_policy(state: Dict[str, Any],
                              actions: List[Tuple[str, Dict]]) -> str:
        # Custom logic here
        return actions[0][0]  # Return first available node

    executor2 = WorkflowExecutor(workflow_graph=graph,
                                 rpc_client=rpc_client,
                                 routing_policy=custom_routing_policy)
