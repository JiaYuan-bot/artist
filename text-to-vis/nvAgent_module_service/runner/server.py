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
from pipeline_functions import preprocess, processor, composer, translator, validator, nv_evaluation

logging.basicConfig(level=logging.INFO)


class FunctionServiceServicer(functions_pb2_grpc.FunctionServiceServicer):
    """Implementation of the FunctionService with all RPC methods."""

    def preprocess(self, request, context):
        """Preprocess based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = preprocess.preprocess(state, config)

            return functions_pb2.PreprocessReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in preprocess: {str(e)}")
            return functions_pb2.PreprocessReply(result_json="{}",
                                                 error=str(e))

    def processor(self, request, context):
        """Processor based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = processor.processor(state, config)

            return functions_pb2.ProcessorReply(result_json=json.dumps(result),
                                                error="")
        except Exception as e:
            logging.error(f"Error in Processor: {str(e)}")
            return functions_pb2.ProcessorReply(result_json="{}", error=str(e))

    def composer(self, request, context):
        """Composer based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = composer.composer(state, config)

            return functions_pb2.ComposerReply(result_json=json.dumps(result),
                                               error="")
        except Exception as e:
            logging.error(f"Error in Composer: {str(e)}")
            return functions_pb2.ComposerReply(result_json="{}", error=str(e))

    def translator(self, request, context):
        """Validator context based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = translator.translator(state, config)

            return functions_pb2.TranslatorReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in Validator: {str(e)}")
            return functions_pb2.TranslatorReply(result_json="{}",
                                                 error=str(e))

    def validator(self, request, context):
        """Validator context based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = validator.validator(state, config)

            return functions_pb2.ValidatorReply(result_json=json.dumps(result),
                                                error="")
        except Exception as e:
            logging.error(f"Error in Validator: {str(e)}")
            return functions_pb2.ValidatorReply(result_json="{}", error=str(e))

    def nv_evaluation(self, request, context):
        """Evaluation entities based on the request."""
        try:
            params = json.loads(request.params_json)
            state = params.get("state", {})
            config = params.get("config", {})
            result = nv_evaluation.nv_evaluation(state, config)

            return functions_pb2.NvEvaluationReply(
                result_json=json.dumps(result), error="")
        except Exception as e:
            logging.error(f"Error in Evaluation: {str(e)}")
            return functions_pb2.NvEvaluationReply(result_json="{}",
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
    logging.info("  - preprocess")
    logging.info("  - processor")
    logging.info("  - composor")
    logging.info("  - validator")
    logging.info("  - evaluation")

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
