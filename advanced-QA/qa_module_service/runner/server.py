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
from pipeline_functions import qa_evaluation, retrieval1, retrieval2, question_to_query, context_question_to_query, context_question_to_answer, rerank

logging.basicConfig(level=logging.INFO)


class FunctionServiceServicer(functions_pb2_grpc.FunctionServiceServicer):
    """Implementation of the FunctionService with all RPC methods."""

    def question_to_query(self, request, context):
        """Preprocess based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = question_to_query.question_to_query(state, config)

            return functions_pb2.QuestionToQueryReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in preprocess: {str(e)}")
            return functions_pb2.QuestionToQueryReply(result_json="{}",
                                                      error=str(e))

    def retrieval1(self, request, context):
        """retrieval1 based on the question_to_query."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = retrieval1.retrieval1(state, config)

            return functions_pb2.Retrieval1Reply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in preprocess: {str(e)}")
            return functions_pb2.Retrieval1Reply(result_json="{}",
                                                 error=str(e))

    def context_question_to_query(self, request, context):
        """Processor based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = context_question_to_query.context_question_to_query(
                state, config)

            return functions_pb2.ContextQuestionToQueryReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in Processor: {str(e)}")
            return functions_pb2.ContextQuestionToQueryReply(result_json="{}",
                                                             error=str(e))

    def retrieval2(self, request, context):
        """retrieval1 based on the question_to_query."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = retrieval2.retrieval2(state, config)

            return functions_pb2.Retrieval2Reply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in preprocess: {str(e)}")
            return functions_pb2.Retrieval2Reply(result_json="{}",
                                                 error=str(e))

    def context_question_to_answer(self, request, context):
        """Composer based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = context_question_to_answer.context_question_to_answer(
                state, config)

            return functions_pb2.ContextQuestionToAnswerReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in Composer: {str(e)}")
            return functions_pb2.ContextQuestionToAnswerReply(result_json="{}",
                                                              error=str(e))

    def rerank(self, request, context):
        """Composer based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = rerank.rerank(state, config)

            return functions_pb2.RerankReply(result_json=json.dumps(result),
                                             error="")
        except Exception as e:
            logging.error(f"Error in Composer: {str(e)}")
            return functions_pb2.RerankReply(result_json="{}", error=str(e))

    def qa_evaluation(self, request, context):
        """Validator context based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = qa_evaluation.qa_evaluation(state, config)

            return functions_pb2.QaEvaluationReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in Validator: {str(e)}")
            return functions_pb2.QaEvaluationReply(result_json="{}",
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
    logging.info("  - question_to_query")
    logging.info("  - context_question_to_query")
    logging.info("  - context_question_to_answer")
    logging.info("  - qa_evaluation")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
        server.stop(0)


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
