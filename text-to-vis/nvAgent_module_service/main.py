import logging
import argparse

from runner import server

import vertexai

if __name__ == "__main__":
    vertexai.init(project='liquid-sylph-476522-r8', location='us-central1')

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
    server.serve(port=args.port)
