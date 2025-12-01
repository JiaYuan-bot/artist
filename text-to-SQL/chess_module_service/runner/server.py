import json
import grpc
import logging
import argparse
from concurrent import futures
from typing import Dict, Any
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_stubs = os.path.join(project_root, 'idl', 'python')
sys.path.append(path_to_stubs)

# Import generated protobuf modules
from idl.python import functions_pb2, functions_pb2_grpc

# Import your existing modules
from pipeline_functions import keyword_extraction, column_filtering, column_selection, context_retrieval, entity_retrieval, evaluation, revision, table_selection, candidate_generation

logging.basicConfig(level=logging.INFO)


class FunctionServiceServicer(functions_pb2_grpc.FunctionServiceServicer):
    """Implementation of the FunctionService with all RPC methods."""

    def candidate_generation(self, request, context):
        """Generate candidates based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = candidate_generation.candidate_generation(state, config)

            return functions_pb2.CandidateGenerationReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in candidate_generation: {str(e)}")
            return functions_pb2.CandidateGenerationReply(result_json="{}",
                                                          error=str(e))

    def column_filtering(self, request, context):
        """Filter columns based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = column_filtering.column_filtering(state, config)

            return functions_pb2.ColumnFilteringReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in column_filtering: {str(e)}")
            return functions_pb2.ColumnFilteringReply(result_json="{}",
                                                      error=str(e))

    def column_selection(self, request, context):
        """Select columns based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = column_selection.column_selection(state, config)

            return functions_pb2.ColumnSelectionReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in column_selection: {str(e)}")
            return functions_pb2.ColumnSelectionReply(result_json="{}",
                                                      error=str(e))

    def context_retrieval(self, request, context):
        """Retrieve context based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = context_retrieval.context_retrieval(state, config)

            return functions_pb2.ContextRetrievalReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in context_retrieval: {str(e)}")
            return functions_pb2.ContextRetrievalReply(result_json="{}",
                                                       error=str(e))

    def entity_retrieval(self, request, context):
        """Retrieve entities based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = entity_retrieval.entity_retrieval(state, config)

            return functions_pb2.EntityRetrievalReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in entity_retrieval: {str(e)}")
            return functions_pb2.EntityRetrievalReply(result_json="{}",
                                                      error=str(e))

    def evaluation(self, request, context):
        """Evaluate based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = evaluation.evaluation(state, config)

            return functions_pb2.EvaluationReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in evaluation: {str(e)}")
            return functions_pb2.EvaluationReply(result_json="{}",
                                                 error=str(e))

    def keyword_extraction(self, request, context):
        """Extract keywords based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = keyword_extraction.keyword_extraction(state, config)

            return functions_pb2.KeywordExtractionReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in keyword_extraction: {str(e)}")
            return functions_pb2.KeywordExtractionReply(result_json="{}",
                                                        error=str(e))

    def revision(self, request, context):
        """Process revision based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = revision.revision(state, config)

            return functions_pb2.RevisionReply(result_json=json.dumps(result),
                                               error="")
        except Exception as e:
            logging.error(f"Error in revision: {str(e)}")
            return functions_pb2.RevisionReply(result_json="{}", error=str(e))

    def table_selection(self, request, context):
        """Select tables based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = table_selection.table_selection(state, config)

            return functions_pb2.TableSelectionReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in table_selection: {str(e)}")
            return functions_pb2.TableSelectionReply(result_json="{}",
                                                     error=str(e))


def serve(port: int):
    """Start the gRPC server."""
    # Create server with thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add the service to the server
    functions_pb2_grpc.add_FunctionServiceServicer_to_server(
        FunctionServiceServicer(), server)
    # Listen on port 50051
    # port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"FunctionService server started on port {port}")
    logging.info("Available RPC methods:")
    logging.info("  - candidate_generation")
    logging.info("  - column_filtering")
    logging.info("  - column_selection")
    logging.info("  - context_retrieval")
    logging.info("  - entity_retrieval")
    logging.info("  - evaluation")
    logging.info("  - keyword_extraction")
    logging.info("  - revision")
    logging.info("  - table_selection")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
        server.stop(0)


# if __name__ == '__main__':
#     serve()

if __name__ == '__main__':
    # Set up basic logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Create a command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run the gRPC FunctionService server.")

    # 2. Add an argument for the port, with a default and help text
    parser.add_argument('--port',
                        type=int,
                        default=50051,
                        help='The port to listen on.')

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # 4. Call the serve function with the specified port
    serve(port=args.port)
