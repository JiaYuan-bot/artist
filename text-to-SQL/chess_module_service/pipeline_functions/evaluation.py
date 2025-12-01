import logging
from typing import Dict, Any

from runner.database_manager import DatabaseManager
from .utils import node_decorator, get_last_node_result


@node_decorator(check_schema_status=False)
def evaluation(
    task: Any,
    tentative_schema: Dict[str, Any],
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates the predicted SQL queries against the ground truth SQL query.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results.
    """
    logging.info("Starting evaluation")

    ground_truth_sql = task.get("SQL", "")
    # Start with the key that's always required

    to_evaluate = {
        "candidate_generation":
        get_last_node_result(execution_history, "candidate_generation")
    }

    # Get the result for the optional key
    revision_result = get_last_node_result(execution_history, "revision")

    # Only add the 'revision' key if its result is not None
    if revision_result is not None:
        to_evaluate["revision"] = revision_result

    result = {}

    for evaluation_for, node_result in to_evaluate.items():
        predicted_sql = "--"
        evaluation_result = {}

        try:
            if node_result["status"] == "success":
                predicted_sql = node_result["SQL"]
                response = DatabaseManager(
                    db_id=task.get("db_id")).compare_sqls(
                        predicted_sql=predicted_sql,
                        ground_truth_sql=ground_truth_sql,
                    )

                evaluation_result.update({
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                })
            else:
                evaluation_result.update({
                    "exec_res": "generation error",
                    "exec_err": node_result["error"],
                })
        except Exception as e:
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
            })

        evaluation_result.update({
            "Question": task.get("question", ""),
            "Evidence": task.get("evidence", ""),
            "GOLD_SQL": ground_truth_sql,
            "PREDICTED_SQL": predicted_sql
        })
        result[evaluation_for] = evaluation_result

    logging.info("Evaluation completed successfully")
    return result
