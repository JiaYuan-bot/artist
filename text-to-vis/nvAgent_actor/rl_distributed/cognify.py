"""
Bayesian Optimization using Optuna for Agentic Workflow LLM Configuration
"""

import optuna
import random
from optuna.samplers import TPESampler
import numpy as np
from typing import Dict, List, Any, Optional
import json
import time
import asyncio
import logging
from dataclasses import dataclass

from workflow_controller.graph import WorkflowGraph, create_workflow_from_config
from workflow_controller.workflow_executor import WorkflowExecutor, FunctionServiceClient
from app.task import Task, NVTask
from nvagent.dataset import Dataset
from pathlib import Path


def setup_logging(log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file. If None, uses timestamp-based filename
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"optimization_{timestamp}_100sample_100trail.log"

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Writing to: {log_file}")

    return log_file


@dataclass
class WorkflowResult:
    """Result from workflow execution"""
    correct: bool
    latency: float
    output: Any
    error: Optional[str] = None


class AgenticWorkflow:
    """
    Agentic workflow that uses different LLMs for different steps
    """

    def __init__(self, config: Dict[str, str]):
        """
        Initialize workflow with model configuration
        
        Args:
            config: Dict like {"planning_model": "gpt-4", "research_model": "claude-sonnet-4.5", ...}
        """
        self.config = config
        self.total_cost = 0.0
        self.step_outputs = {}

    def select_action(self, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy"""
        return random.choice(available_actions)

    async def collect_episode_experience(self, task: NVTask,
                                         workflow_executor: WorkflowExecutor):
        """Collect experience from a single episode"""
        episode_trajectory_raw = []
        success = True
        tracer = workflow_executor.tracer
        state = tracer.start_execution(task=task, session_id=f"actor_{1}")
        await tracer.execute_node("START")  # Add await
        current_state_rl = tracer.current_state.to_rl_state()

        while not tracer.is_execution_complete():
            prev_state_rl = current_state_rl
            available_actions = tracer.get_available_actions()
            if not available_actions:
                success = False
                break

            action_nodes = [a[0] for a in available_actions]
            selected_action = self.select_action(
                action_nodes)  # Fixed: removed extra args

            episode_trajectory_raw.append({
                'state_graph_part': prev_state_rl,
                'action': selected_action,
                'available_actions': action_nodes
            })

            result = await tracer.execute_node(selected_action)  # Add await
            if not result.success:
                success = False
                break
            current_state_rl = tracer.current_state.to_rl_state()

        if not success:
            return {"is_correct": False, "score": 0.0}

        # Process trajectory into experiences
        overall_episode_successful, score = tracer.nv_execution_success()
        tracer.current_state.success = overall_episode_successful

        path = tracer.current_state.executed_nodes
        nodes_exec_time = []

        execution_history = tracer.current_state.accumulated_outputs.get(
            'nv_evaluation')["keys"]["execution_history"]

        for i in range(len(path)):
            node_exec_time = tracer.current_state.execution_history[
                i].execution_time
            if i - 1 >= 0 and i != (len(path) - 1):
                cascade = execution_history[i - 1].get('cascade', 0)
            else:
                cascade = 0
            nodes_exec_time.append(
                f"{path[i]}({node_exec_time:.2f}s) {cascade}")

        logging.info(
            f"Task Id: {task.id} Path length: {len(path)}, "
            f"Execution Time: {tracer.current_state.total_execution_time:.2f}s, "
            f"Executed Nodes: {nodes_exec_time}\n"
            f"Result: {overall_episode_successful}, score: {score}")

        return {"is_correct": overall_episode_successful, "score": score}


class WorkflowOptimizer:
    """
    Bayesian Optimization for workflow configuration using Optuna
    """

    def __init__(self,
                 workflow_steps: Dict[str, List[str]],
                 test_cases: List[NVTask],
                 optimization_mode: str = "constrained"):
        """
        Initialize the workflow optimizer
        
        Args:
            workflow_steps: Dict mapping step names to available LLM choices
                           e.g., {"planning": ["gpt-4", "claude-sonnet-4.5"], ...}
            test_cases: List of test cases for evaluation
            optimization_mode: "single" (single objective), "multi" (multi-objective), 
                              or "constrained" (with constraints)
        """
        self.workflow_steps = workflow_steps
        self.test_cases = test_cases
        self.optimization_mode = optimization_mode
        self.eval_cache = {}

    async def objective_single(self, trial: optuna.Trial) -> float:
        """
        Single objective: Maximize accuracy while penalizing cost and latency
        
        Returns higher values for better configurations
        """
        # Sample model for each step
        config = {}
        for step_name, model_choices in self.workflow_steps.items():
            if len(model_choices) > 0:
                config[f"{step_name}"] = trial.suggest_categorical(
                    f"{step_name}", model_choices)

        # Evaluate configuration
        metrics = await self.evaluate_configuration(config, trial.number)

        # Composite score: maximize accuracy, minimize latency
        score = (
            metrics["accuracy"] * 100  # Scale accuracy (0-1) to 0-100
            - metrics["latency"] * 2  # Penalize latency
        )

        # Log metrics for analysis
        trial.set_user_attr("accuracy", metrics["accuracy"])
        trial.set_user_attr("latency", metrics["latency"])

        return score

    async def objective_multi(self,
                              trial: optuna.Trial) -> tuple[float, float]:
        """
        Multi-objective: Optimize accuracy and latency simultaneously
        
        Returns tuple of (accuracy, latency)
        Optuna will find Pareto-optimal solutions
        """
        # Sample model for each step
        config = {}
        for step_name, model_choices in self.workflow_steps.items():
            if len(model_choices) > 0:
                config[f"{step_name}"] = trial.suggest_categorical(
                    f"{step_name}", model_choices)

        # Evaluate configuration
        metrics = await self.evaluate_configuration(config, trial.number)

        # Return multiple objectives
        # Optuna maximizes by default, so negate latency to minimize it
        return (
            metrics["accuracy"],  # Maximize
            -metrics["latency"]  # Minimize (negated)
        )

    def objective_constrained(self, trial: optuna.Trial) -> float:
        """
        Constrained optimization: Maximize accuracy subject to latency constraints
        """
        # Sample model for each step
        config = {}
        for step_name, model_choices in self.workflow_steps.items():
            if len(model_choices) > 0:
                config[f"{step_name}"] = trial.suggest_categorical(
                    f"{step_name}", model_choices)

        print("Starting evaluate configuration")
        # Evaluate configuration
        metrics = asyncio.run(self.evaluate_configuration(
            config, trial.number))
        # Add constraints
        # If constraints violated, prune this trial
        MAX_LATENCY = 100.0  # Maximum 10 seconds
        # MIN_ACCURACY = 0.6667
        # MIN_ACCURACY = 0.6600
        MIN_ACCURACY = 0.6350

        # if metrics["latency"] > MAX_LATENCY:
        #     raise optuna.TrialPruned(
        #         f"Latency {metrics['latency']:.2f}s exceeds limit {MAX_LATENCY}s"
        #     )

        if metrics["accuracy"] < MIN_ACCURACY:
            logging.error(
                f"accuracy {metrics['accuracy']:.2f} lower than limit {MIN_ACCURACY}"
            )
            raise optuna.TrialPruned(
                f"accuracy {metrics['accuracy']:.2f} lower than limit {MIN_ACCURACY}"
            )

        # Log metrics
        trial.set_user_attr("latency", metrics["latency"])
        trial.set_user_attr("accuracy", metrics["accuracy"])

        # # Return accuracy to maximize
        # return metrics["accuracy"]

        # Return latencuy to minimize
        return metrics["latency"]

    async def evaluate_configuration(
            self,
            config: Dict[str, str],
            trial_num: int,
            max_concurrent: int = 500) -> Dict[str, float]:
        """
        Evaluate a specific LLM configuration with limited concurrency
        
        Args:
            config: Dict mapping step names to selected models
            trial_num: Trial number for logging
            max_concurrent: Maximum number of concurrent executions
            
        Returns:
            Dict of metric scores
        """
        # Check cache
        config_key = json.dumps(config, sort_keys=True)
        if config_key in self.eval_cache:
            logging.info(f"[Trial {trial_num}] Using cached result")
            return self.eval_cache[config_key]

        logging.info(f"[Trial {trial_num}] Evaluating: {config}")

        function_service_client_ports = list(range(40151, 40201))

        rpc_clients = []
        for port in function_service_client_ports:
            rpc_client = FunctionServiceClient(port=port)
            rpc_clients.append(rpc_client)

        graph = create_workflow_from_config(config=config)

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(idx, test_case):
            """Run workflow with semaphore to limit concurrency"""
            async with semaphore:
                rpc_client = rpc_clients[idx % len(rpc_clients)]
                workflow_executor = WorkflowExecutor(workflow_graph=graph,
                                                     rpc_client=rpc_client)
                workflow = AgenticWorkflow(config)
                return await self.run_workflow(workflow, test_case,
                                               workflow_executor)

        # Create tasks
        tasks = [
            run_with_semaphore(idx, test_case)
            for idx, test_case in enumerate(self.test_cases)
        ]

        # Execute with limited concurrency
        logging.info(f"[Trial {trial_num}] Running {len(tasks)} test cases "
                     f"(max {max_concurrent} concurrent)...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results (same as before)
        valid_results = []
        total_latency = 0.0

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(
                    f"[Trial {trial_num}] Test case {idx+1} failed: {result}")
                valid_results.append(
                    WorkflowResult(correct=False,
                                   latency=0.0,
                                   output=None,
                                   error=str(result)))
            else:
                valid_results.append(result)
                total_latency += result.latency
                logging.debug(
                    f"[Trial {trial_num}] Test case {idx+1}/{len(self.test_cases)}: "
                    f"correct={result.correct}, latency={result.latency:.2f}s")

        # Compute metrics
        n_correct = sum(1 for r in valid_results if r.correct)
        metrics = {
            "accuracy": n_correct / len(valid_results),
            "latency": total_latency / len(valid_results),
        }

        logging.info(
            f"[Trial {trial_num}] Results: accuracy={metrics['accuracy']:.3f}, "
            f"latency={metrics['latency']:.2f}s")

        # Cache result
        self.eval_cache[config_key] = metrics

        return metrics

    async def run_workflow(
            self, workflow: AgenticWorkflow, test_case: NVTask,
            workflow_executor: WorkflowExecutor) -> WorkflowResult:
        """
        Execute workflow and return structured result
        """
        start_time = time.time()

        try:
            output = await workflow.collect_episode_experience(
                test_case, workflow_executor)
            latency = time.time() - start_time

            # Evaluate correctness
            is_correct = output.get('is_correct')

            return WorkflowResult(correct=is_correct,
                                  latency=latency,
                                  output=output)

        except Exception as e:
            print(f"Error in run_workflow: {e}", exc_info=True)
            return WorkflowResult(correct=False,
                                  latency=time.time() - start_time,
                                  output=None,
                                  error=str(e))

    def optimize(self,
                 n_trials: int = 30,
                 direction: str = "maximize") -> Dict[str, Any]:
        """
        Run Bayesian Optimization using Optuna
        
        Args:
            n_trials: Number of configurations to evaluate
            direction: "maximize" or "minimize" (for single objective)
            
        Returns:
            Dict with best configuration and optimization results
        """
        print("\n" + "=" * 70)
        print("STARTING BAYESIAN OPTIMIZATION WITH OPTUNA")
        print("=" * 70)
        print(f"Mode: {self.optimization_mode}")
        print(f"Test cases: {len(self.test_cases)}")
        print(f"Trials: {n_trials}\n")

        # Create study based on optimization mode
        if self.optimization_mode == "single":
            study = optuna.create_study(direction=direction,
                                        sampler=TPESampler(seed=42),
                                        study_name="workflow_optimization")
            objective_func = self.objective_single

        elif self.optimization_mode == "multi":
            study = optuna.create_study(
                directions=["maximize", "maximize"],  # accuracy, -latency
                sampler=TPESampler(seed=42, multivariate=True),
                study_name="workflow_multi_objective")
            objective_func = self.objective_multi

        elif self.optimization_mode == "constrained":
            study = optuna.create_study(
                direction="minimize",  # minimize, maximize
                sampler=TPESampler(seed=42),
                study_name="workflow_constrained")
            objective_func = self.objective_constrained

        else:
            raise ValueError(
                f"Unknown optimization mode: {self.optimization_mode}")

        # Run optimization with async support
        # Optuna handles async objectives automatically
        study.optimize(objective_func,
                       n_trials=n_trials,
                       show_progress_bar=True)

        # Get results
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        if self.optimization_mode == "multi":
            # Multi-objective: show Pareto front
            print(
                f"\nFound {len(study.best_trials)} Pareto-optimal solutions:")
            for i, trial in enumerate(study.best_trials[:5]):  # Show top 5
                print(f"\nSolution {i+1}:")
                print(f"  Configuration: {trial.params}")
                print(f"  Accuracy: {trial.values[0]:.3f}")
                print(f"  Latency: {-trial.values[1]:.2f}s")

            best_config = study.best_trials[0].params

        else:
            # Single objective or constrained
            best_trial = study.best_trial
            print(f"\nBest Configuration:")
            for param, value in best_trial.params.items():
                print(f"  {param}: {value}")

            print(f"\nBest Score: {best_trial.value:.4f}")

            if self.optimization_mode == "single":
                print(
                    f"  Accuracy: {best_trial.user_attrs.get('accuracy', 'N/A'):.3f}"
                )
                print(
                    f"  Latency: {best_trial.user_attrs.get('latency', 'N/A'):.2f}s"
                )
            else:  # constrained
                print(
                    f"  Latency: {best_trial.user_attrs.get('latency', 'N/A'):.2f}s"
                )

            best_config = best_trial.params

        # Save results
        results = {
            "best_config": best_config,
            "study": study,
            "n_trials": len(study.trials),
            "optimization_mode": self.optimization_mode
        }

        return results


def run_cognify():
    log_file = setup_logging("cognify_optimization_300sample_32trail.log")

    # Define your workflow structure
    workflow_steps = {
        "preprocess": [],
        "processor":
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "composer":
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "translator": [],
        "validator":
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "nv_evaluation": []
    }

    # test, eva
    ds = Dataset(folder=Path("/home/jiayuan/nl2sql/nvAgent/visEval_dataset"),
                 table_type="test")
    dataset = ds.load_dataset()
    dataset = dataset[0:300]
    # print(dataset[0])

    # Constrained optimization
    print("\n\n### CONSTRAINED OPTIMIZATION ###")
    optimizer_constrained = WorkflowOptimizer(workflow_steps,
                                              dataset,
                                              optimization_mode="constrained")
    results_constrained = optimizer_constrained.optimize(n_trials=32)

    # Save best configuration
    with open("best_config.json", "w") as f:
        json.dump(results_constrained["best_config"], f, indent=2)

    print("\nBest configuration saved to best_config.json")


if __name__ == "__main__":
    run_cognify()
