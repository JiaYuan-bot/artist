import sklearn  # Add this line at the very top
import torch.cuda
import logging
import asyncio

from workflow_controller.workflow_executor import WorkflowExecutor, FunctionServiceClient
# from rl_distributed.learner import run_learner
from rl_distributed.learner_with_p_buffer import run_sync_learner
from rl_distributed.experience_collector import start_collector_server
from rl_distributed.learner_with_p_buffer import sample_experience

if __name__ == "__main__":
    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    torch.autograd.set_detect_anomaly(True)

    # sample_experience()

    # Example usage
    from workflow_controller.graph import create_example_nvagent_workflow_augmented

    # Create workflow and dataset
    graph = create_example_nvagent_workflow_augmented()
    run_sync_learner(workflow_graph=graph, device='cuda')

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description='Simple Experience Collector Learner')
#     parser.add_argument('--port', type=int, default=50201, help='Server port')
#     parser.add_argument('--buffer-capacity',
#                         type=int,
#                         default=50000000,
#                         help='Maximum buffer capacity')
#     parser.add_argument('--buffer-save-path',
#                         type=str,
#                         default='experience_buffer_3.pkl',
#                         help='Path to save/load buffer')
#     parser.add_argument('--max-workers',
#                         type=int,
#                         default=1000,
#                         help='Maximum concurrent RPC workers')

#     args = parser.parse_args()

#     start_collector_server(port=args.port,
#                            buffer_capacity=args.buffer_capacity,
#                            buffer_save_path=args.buffer_save_path,
#                            max_workers=args.max_workers)
