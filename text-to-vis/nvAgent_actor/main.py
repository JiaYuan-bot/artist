import logging
import asyncio

from workflow_controller.workflow_executor import WorkflowExecutor, FunctionServiceClient
# from rl_distributed.actor_model_free import main
from rl_distributed.actor_model_sync import sync_nvAgent
from rl_distributed.actor_test import test_nvAgent, encoder_test
from rl_distributed.cognify import run_cognify

from app.task import Task

if __name__ == "__main__":
    # Setup logging, etc.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # encoder_test()
    # Start the single event loop to run everything
    asyncio.run(test_nvAgent())
    # run_cognify()
