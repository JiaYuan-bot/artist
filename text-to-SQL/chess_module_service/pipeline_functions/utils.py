from functools import wraps
from typing import Dict, List, Any, Callable
from runner.database_manager import DatabaseManager


def node_decorator(check_schema_status: bool = False) -> Callable:
    """
    A decorator to add logging and error handling to pipeline node functions.

    Args:
        check_schema_status (bool, optional): Whether to check the schema status. Defaults to False.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(
            state: Dict[str, Any],
            config: Dict[str, Any],
        ) -> Dict[str, Any]:
            node_name = func.__name__
            result = {"node_type": node_name}

            try:
                task = state["keys"]["task"]
                tentative_schema = state["keys"]["tentative_schema"]
                execution_history = state["keys"]["execution_history"]

                if not tentative_schema:
                    tentative_schema = DatabaseManager(
                        db_id=task.get("db_id")).get_db_schema()
                    state["keys"]["tentative_schema"] = tentative_schema

                output = func(task, tentative_schema, execution_history,
                              config)
                if "tentative_schema" in output:
                    tentative_schema = output["tentative_schema"]
                    state["keys"]["tentative_schema"] = tentative_schema
                result.update(output)
                if check_schema_status:
                    result.update(missings_status(task, tentative_schema))
                result["status"] = "success"
            except Exception as e:
                result.update({
                    "status": "error",
                    "error": f"{type(e)}: <{e}>",
                })

            execution_history.append(result)

            return state

        return wrapper

    return decorator


def get_last_node_result(execution_history: List[Dict[str, Any]],
                         node_type: str) -> Dict[str, Any]:
    """
    Retrieves the last result for a specific node type from the execution history.

    Args:
        execution_history (List[Dict[str, Any]]): The execution history.
        node_type (str): The type of node to look for.

    Returns:
        Dict[str, Any]: The result of the last node of the specified type, or None if not found.
    """
    for node in reversed(execution_history):
        if node["node_type"] == node_type:
            return node
    return None


def missings_status(task: Any,
                    tentative_schema: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Checks for missing tables and columns in the tentative schema.

    Args:
        task (Any): The current task.
        tentative_schema (Dict[str, List[str]]): The tentative schema.

    Returns:
        Dict[str, Any]: A dictionary with the status of missing tables and columns.
    """
    ground_truth_sql = task.get("SQL", "")
    correct_columns = DatabaseManager(
        db_id=task.get("db_id")).get_sql_columns_dict(sql=ground_truth_sql)
    missing_tables = []
    missing_columns = []

    for table_name, cols in correct_columns.items():
        for col in cols:
            selected_table = [
                table for table in tentative_schema.keys()
                if table.lower() == table_name.lower()
            ]
            if not selected_table:
                if table_name not in missing_tables:
                    missing_tables.append(table_name)
            else:
                selected_table = selected_table[0]
                if col.lower() not in [
                        selected_col.lower()
                        for selected_col in tentative_schema[selected_table]
                ]:
                    missing_columns.append(f"'{table_name}'.'{col}'")

    status = {
        "missing_table_status":
        "success" if not missing_tables else "missing_table",
        "missing_tables": missing_tables,
        "missing_column_status":
        "success" if not missing_columns else "missing_column",
        "missing_columns": missing_columns,
        "correct_columns": correct_columns,
    }
    return status


def add_columns_to_tentative_schema(
        tentative_schema: Dict[str, List[str]],
        selected_columns: Dict[str, List[str]]) -> None:
    """
    Adds columns to the tentative schema based on selected columns.

    Args:
        tentative_schema (Dict[str, List[str]]): The tentative schema.
        selected_columns (Dict[str, List[str]]): The selected columns to add.
    """
    for table_name, columns in selected_columns.items():
        target_table_name = next((t for t in tentative_schema.keys()
                                  if t.lower() == table_name.lower()), None)
        if target_table_name:
            for column in columns:
                if column.lower() not in [
                        c.lower() for c in tentative_schema[target_table_name]
                ]:
                    tentative_schema[target_table_name].append(column)
        else:
            tentative_schema[table_name] = columns
