from typing import Dict, List, Optional, Any, Tuple
import time
import asyncio
import grpc.aio as aio_grpc
import logging
import grpc
import json
import numpy as np
from dataclasses import asdict

# Import from workflow_graph module
from .graph import Node, WorkflowGraph, NodeType
from workflow_controller.workflow_states import WorkflowState, ExecutionResult

# Import generated protobuf modules
from idl.python import functions_pb2, functions_pb2_grpc

from app.task import Task


class FunctionServiceClient:
    """Client for the unified FunctionService that handles all RPC calls."""

    def __init__(self, host='localhost', port=50051):
        """
        Initialize the FunctionService client.
        
        Args:
            host: The server host
            port: The server port
        """
        self.channel = aio_grpc.insecure_channel(f'{host}:{port}')
        self.stub = functions_pb2_grpc.FunctionServiceStub(self.channel)

        # Map function names to their corresponding RPC methods and request/reply classes
        self.function_map = {
            ### nl2sql ###
            'candidate_generation': {
                'method': self.stub.candidate_generation,
                'request_class': functions_pb2.CandidateGenerationReq,
                'reply_class': functions_pb2.CandidateGenerationReply
            },
            'column_filtering': {
                'method': self.stub.column_filtering,
                'request_class': functions_pb2.ColumnFilteringReq,
                'reply_class': functions_pb2.ColumnFilteringReply
            },
            'column_selection': {
                'method': self.stub.column_selection,
                'request_class': functions_pb2.ColumnSelectionReq,
                'reply_class': functions_pb2.ColumnSelectionReply
            },
            'context_retrieval': {
                'method': self.stub.context_retrieval,
                'request_class': functions_pb2.ContextRetrievalReq,
                'reply_class': functions_pb2.ContextRetrievalReply
            },
            'entity_retrieval': {
                'method': self.stub.entity_retrieval,
                'request_class': functions_pb2.EntityRetrievalReq,
                'reply_class': functions_pb2.EntityRetrievalReply
            },
            'evaluation': {
                'method': self.stub.evaluation,
                'request_class': functions_pb2.EvaluationReq,
                'reply_class': functions_pb2.EvaluationReply
            },
            'keyword_extraction': {
                'method': self.stub.keyword_extraction,
                'request_class': functions_pb2.KeywordExtractionReq,
                'reply_class': functions_pb2.KeywordExtractionReply
            },
            'revision': {
                'method': self.stub.revision,
                'request_class': functions_pb2.RevisionReq,
                'reply_class': functions_pb2.RevisionReply
            },
            'table_selection': {
                'method': self.stub.table_selection,
                'request_class': functions_pb2.TableSelectionReq,
                'reply_class': functions_pb2.TableSelectionReply
            },

            ### nvAgent ###
            'preprocess': {
                'method': self.stub.preprocess,
                'request_class': functions_pb2.PreprocessReq,
                'reply_class': functions_pb2.PreprocessReply
            },
            'processor': {
                'method': self.stub.processor,
                'request_class': functions_pb2.ProcessorReq,
                'reply_class': functions_pb2.ProcessorReply
            },
            'composer': {
                'method': self.stub.composer,
                'request_class': functions_pb2.ComposerReq,
                'reply_class': functions_pb2.ComposerReply
            },
            'translator': {
                'method': self.stub.translator,
                'request_class': functions_pb2.TranslatorReq,
                'reply_class': functions_pb2.TranslatorReply
            },
            'validator': {
                'method': self.stub.validator,
                'request_class': functions_pb2.ValidatorReq,
                'reply_class': functions_pb2.ValidatorReply
            },
            'nv_evaluation': {
                'method': self.stub.nv_evaluation,
                'request_class': functions_pb2.NvEvaluationReq,
                'reply_class': functions_pb2.NvEvaluationReply
            },
        }

    async def call(self,
                   function_name: str,
                   node: Node,
                   inputs: Dict[str, Any],
                   timeout: float = 2000.0) -> Dict[str, Any]:
        """
        Call an RPC function by name with gRPC native timeout.
        
        Args:
            function_name: Name of the function to call
            node: Node object containing metadata
            inputs: Input parameters for the function
            timeout: Timeout in seconds (default: 30.0)
            
        Returns:
            Dictionary containing the function result, or {} if timeout occurs
            
        Raises:
            ValueError: If function_name is not registered
            Exception: If the RPC call fails (non-timeout errors)
        """
        if function_name not in self.function_map:
            raise ValueError(
                f"Unknown function: {function_name}. Available functions: {list(self.function_map.keys())}"
            )

        try:
            # Get function info
            func_info = self.function_map[function_name]
            method = func_info['method']
            request_class = func_info['request_class']

            # Prepare parameters
            params = {
                **inputs,  # Include all inputs
            }

            # Create the request
            request = request_class(params_json=json.dumps(params))

            # Make the RPC call with gRPC timeout
            response = await method(request, timeout=timeout)

            # Check for errors
            if hasattr(response, 'error') and response.error:
                raise Exception(
                    f"Server error in {function_name}: {response.error}")

            # Parse and return the result
            result = json.loads(response.result_json)
            # logging.info(f"Successfully called {function_name}")
            return result

        except aio_grpc.AioRpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logging.warning(
                    f"RPC call to {function_name} timed out after {timeout}s, returning None"
                )
                return None
            logging.error(f"RPC failed for {function_name}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error calling {function_name}: {e}")
            raise


class WorkflowTracer:
    """Traces and executes workflow based on RL routing decisions"""

    def __init__(self,
                 workflow_graph: WorkflowGraph,
                 rpc_client: FunctionServiceClient,
                 logger: Optional[logging.Logger] = None):
        self.graph = workflow_graph
        self.rpc_client = rpc_client
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Execution state
        self.current_state: Optional[WorkflowState] = None
        self.execution_session_id: Optional[str] = None

    def start_execution(self, task: Task, session_id: str) -> WorkflowState:
        """Initialize a new workflow execution"""
        self.execution_session_id = session_id
        self.current_state = WorkflowState(task=task,
                                           current_node_id="",
                                           executed_nodes=[],
                                           execution_history=[],
                                           accumulated_outputs={})
        # self.logger.info(f"Started workflow execution: {session_id}")
        return self.current_state

    async def execute_node(self, node_id: str) -> ExecutionResult:
        """Execute a single node based on RL decision"""
        if not self.current_state:
            raise ValueError("No active execution session")

        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in workflow graph")

        node = self.graph.nodes[node_id]

        result = ExecutionResult(node_id=node_id)

        if node.node_type == NodeType.END or node.node_type == NodeType.START:
            result.end_time = time.time()
            self._update_state(node_id, result)
            return result

        try:
            # Prepare inputs from accumulated outputs
            inputs = self._prepare_node_inputs(node)
            # Make RPC call
            # self.logger.info(f"Executing node {node_id} with inputs: {inputs}")
            output = await self.rpc_client.call(
                function_name=node.function_name.value,
                node=node,
                inputs=inputs)

            # Update result
            result.output = output
            result.end_time = time.time()

            if not output:
                result.success = False

            # self.logger.info(
            #     f"Node {node_id} executed successfully in {result.execution_time:.3f}s, output: {output}"
            # )

        except Exception as e:
            result.error = str(e)
            result.end_time = time.time()
            result.success = False
            self.logger.error(f"Node {node_id} execution failed: {e}")

        # Update workflow state
        self._update_state(node_id, result)

        return result

    def _prepare_node_inputs(self, node: Node) -> Dict[str, Any]:
        """Prepare inputs for node execution"""
        if self.current_state.current_node_id == "START":
            inputs = {
                "state": {
                    "keys": {
                        "task": asdict(self.current_state.task),
                        "tentative_schema": {},
                        "execution_history": []
                    }
                },
                "config": {
                    "llm_config": {
                        "model": node.config.model,
                        "temperature": node.config.temperature,
                        "template_name": node.config.template_name,
                        "parser_name": node.config.parser_name
                    },
                    node.function_name.value: node.config.custom_params
                }
            }
        else:
            inputs = {
                "state":
                self.current_state.execution_history[-1].output
                if len(self.current_state.execution_history) > 0 else None,
                "config": {
                    "llm_config": {
                        "model": node.config.model,
                        "temperature": node.config.temperature,
                        "template_name": node.config.template_name,
                        "parser_name": node.config.parser_name
                    },
                    node.function_name.value: node.config.custom_params
                }
            }

        return inputs

    def _update_state(self, node_id: str, result: ExecutionResult):
        """Update workflow state after node execution"""
        # Update execution history
        self.current_state.execution_history.append(result)
        self.current_state.executed_nodes.append(node_id)

        # Update accumulated outputs
        if result.output:
            self.current_state.accumulated_outputs[node_id] = result.output

        # Update total execution time
        if result.execution_time:
            self.current_state.total_execution_time += result.execution_time

        # Update current node
        self.current_state.current_node_id = node_id

    def get_available_actions(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get available next nodes (actions) for current state"""
        if not self.current_state:
            return []

        return self.graph.get_node_successors(
            self.current_state.current_node_id)

    def is_execution_complete(self) -> bool:
        """Check if workflow execution has reached END node"""
        return self.current_state and self.current_state.current_node_id == "END"

    def get_execution_trace(self) -> Dict[str, Any]:
        """Get complete execution trace for analysis"""
        if not self.current_state:
            return {}

        return {
            "session_id":
            self.execution_session_id,
            "executed_path":
            self.current_state.executed_nodes,
            "total_time":
            self.current_state.total_execution_time,
            "node_executions": [{
                "node_id": result.node_id,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "output": result.output,
                "error": result.error
            } for result in self.current_state.execution_history],
            "final_output":
            self.current_state.accumulated_outputs.get(
                self.current_state.executed_nodes[-2]
                if len(self.current_state.executed_nodes) > 1 else None),
            "request_features":
            self.current_state.request_features
        }

    def calculate_execution_metrics(self) -> Dict[str, float]:
        """Calculate metrics for RL reward calculation"""
        if not self.current_state:
            return {}

        metrics = {
            "total_time": self.current_state.total_execution_time,
            "node_count": len(self.current_state.executed_nodes)
        }

        evaluation_result = self.current_state.accumulated_outputs.get(
            "evaluation")['keys'].get("execution_history")[-1]

        revision_output_result = None
        if evaluation_result.get("revision"):
            revision_output_result = evaluation_result.get("revision").get(
                "exec_err")

        candidate_generation_output_result = None
        if evaluation_result.get("candidate_generation"):
            candidate_generation_output_result = evaluation_result.get(
                "candidate_generation").get("exec_err")
        metrics[
            "output_result"] = revision_output_result == "--" or candidate_generation_output_result == "--"

        return metrics

    def execution_success(self) -> bool:
        """Calculate metrics for RL reward calculation"""
        if not self.current_state:
            return False

        evaluation_result = self.current_state.accumulated_outputs.get(
            "evaluation")['keys'].get("execution_history")[-1]

        revision_output_result = None
        if evaluation_result.get("revision"):
            revision_output_result = evaluation_result.get("revision").get(
                "exec_err")

        candidate_generation_output_result = None
        if evaluation_result.get("candidate_generation"):
            candidate_generation_output_result = evaluation_result.get(
                "candidate_generation").get("exec_err")
        return revision_output_result == "--" or candidate_generation_output_result == "--"

    def nv_execution_success(self) -> Tuple[bool, int]:
        """Calculate metrics for RL reward calculation"""
        if not self.current_state:
            return False

        nv_evaluation_result = self.current_state.accumulated_outputs.get(
            "nv_evaluation")['keys'].get("execution_history")[-1]
        return nv_evaluation_result.get("pass"), nv_evaluation_result.get(
            "readability_score")
