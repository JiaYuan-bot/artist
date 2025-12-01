import re
import json
from typing import Dict
import sqlglot
import logging
from functools import wraps
from typing import Dict, List, Any, Callable


def node_decorator() -> Callable:
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
                execution_history = state["keys"]["execution_history"]

                output = func(task, execution_history, config)

                result.update(output)

                result["status"] = "success"
            except Exception as e:
                logging.error(f"{node_name} failed. Error:{e}")
                result.update({
                    "status": "error",
                    "error": f"{type(e)}: <{e}>",
                })

            execution_history.append(result)

            return state

        return wrapper

    return decorator


def get_last_node_result(
        execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(execution_history) == 0:
        return {}

    return execution_history[-1]


def get_node_result(execution_history: List[Dict[str, Any]],
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


def parse_response(response: str) -> Dict:

    result = {
        "filtered_schema": {},
        "new_schema": "",
        "augmented_explanation": "",
        "query_difficulty": "",
    }

    #  Filtered Schema
    filtered_schema_match = re.search(
        r'【Filtered Schema】\n(.*?)\n\n【New Schema】', response, re.DOTALL)
    if filtered_schema_match:
        result["filtered_schema"] = filtered_schema_match.group(1).strip()

    #  database Schema
    new_schema_match = re.search(
        r'【New Schema】\n(.*?)\n\n【Augmented Explanation】', response, re.DOTALL)
    if new_schema_match:
        result["new_schema"] = new_schema_match.group(1).strip()

    #  Format Explanation
    augmented_explanation_match = re.search(
        r'【Augmented Explanation】\n(.*?)\n\n【Classification】', response,
        re.DOTALL)
    if augmented_explanation_match:
        result["augmented_explanation"] = augmented_explanation_match.group(
            1).strip()

    # Query Difficulty
    query_difficulty_match = re.search(r'【Classification】\n(\w+)', response)
    if query_difficulty_match:
        result["query_difficulty"] = query_difficulty_match.group(1).strip()

    return result


def has_order_by(vql):
    return bool(re.search(r'\bORDER\s+BY\b', vql, re.IGNORECASE))


def validate_select_order(vql: str) -> bool:

    match = re.search(r'Visualize\s+([\w\s]+)\s+SELECT\s+(.*?)\s+FROM', vql,
                      re.IGNORECASE | re.DOTALL)
    if not match:
        return False

    vis_type = match.group(1).upper().strip()
    select_columns = [col.strip() for col in match.group(2).split(',')]

    if vis_type in ['BAR', 'PIE', 'LINE', 'SCATTER']:
        return len(select_columns) == 2
    elif vis_type in ['STACKED BAR', 'GROUPED LINE', 'GROUPED SCATTER']:
        return len(select_columns) == 3
    else:
        return False


def add_group_by(sql, new_group_by_column):
    parsed = sqlglot.parse_one(sql)

    group_by = parsed.find(sqlglot.expressions.Group)

    if group_by:

        group_by_columns = group_by.expressions

        if not any(col.name == new_group_by_column
                   for col in group_by_columns):

            group_by_columns.append(
                sqlglot.exp.Column(this=new_group_by_column))
    else:
        group_by = sqlglot.exp.Group(
            expressions=[sqlglot.exp.Column(this=new_group_by_column)])
        parsed.set("group", group_by)

    return parsed.sql()


def show_svg(plt, svg_name: str):
    """Show a plot as a SVG inline."""
    from io import StringIO
    f = StringIO()
    plt.savefig(f, format="svg")
    if svg_name:
        plt.savefig(f"{svg_name}")
    svg_content = f.getvalue()
    plt.close()

    return svg_content


def parse_vql_from_string(response: str):
    vql_matches = re.findall(r'Visualize\s+.*', response,
                             re.IGNORECASE | re.MULTILINE)
    if vql_matches:
        return vql_matches[-1].strip()
    else:

        return ''


def parse_code_from_string(response: str):
    code_matches = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)

    if code_matches:
        return code_matches[-1].strip()
    else:
        return None


def is_valid_date(date_str):
    if (not isinstance(date_str, str)):
        return False
    date_str = date_str.split()[0]
    if len(date_str) != 10:
        return False
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date_str):
        year, month, day = map(int, date_str.split('-'))
        if year < 1 or month < 1 or month > 12 or day < 1 or day > 31:
            return False
        else:
            return True
    else:
        return False


def is_valid_date_column(col_value_lst):
    for col_value in col_value_lst:
        if not is_valid_date(col_value):
            return False
    return True


def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False


def extract_world_info(message_dict: dict):
    info_dict = {}
    info_dict['idx'] = message_dict.get('idx', 0)
    info_dict['db_id'] = message_dict['db_id']
    info_dict['query'] = message_dict['query']
    info_dict['difficulty'] = message_dict.get('difficulty', '')
    info_dict['ground_truth'] = message_dict.get('ground_truth', '')
    info_dict['send_to'] = message_dict.get('send_to', '')
    return info_dict


def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"load json file from {path}")
        return json.load(f)


def show_svg(plt, svg_name: str):
    """Show a plot as a SVG inline."""
    from io import StringIO
    f = StringIO()
    plt.savefig(f, format="svg")
    if svg_name:
        plt.savefig(f"{svg_name}")
    svg_content = f.getvalue()
    plt.close()

    return svg_content
