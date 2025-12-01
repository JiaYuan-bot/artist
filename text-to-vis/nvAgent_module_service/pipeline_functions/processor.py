import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path
from tqdm import trange

from llm.models import gemini_api_call_with_config
from .utils import node_decorator, get_last_node_result
from .utils import parse_response, extract_world_info, is_email, is_valid_date_column
from .const import *
"""
Get database description, extract relative tables & columns, generate augmented explanation and query difficulty
"""


@node_decorator()
def processor(
    task: Any,
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
):
    """
    :param task:{
                    "db_id": database_name,
                    "query": user_query,
                    "tables": original tables information(table name)
                }
    :return: extracted database schema 
    """
    logging.info("Starting processor")

    query, db_id = task.get('query'), task.get('db_id')

    db_schema = get_last_node_result(
        execution_history)["old_schema"] if get_last_node_result(
            execution_history) else ''

    llm_config = config.get("llm_config", {})
    model_name = llm_config.get("model", "gemini-2.5-pro")

    # # without processor
    # message['old_schema'] = db_schema
    # message['send_to'] = COMPOSER_NAME
    # return

    try:
        result = process(db_id=db_id,
                         query=query,
                         db_schema=db_schema,
                         model_name=model_name)
    except Exception as e:
        logging.error(e)
        result = {
            "filtered_schema": {},
            "new_schema": db_schema,
            "augmented_explanation": "",
            "query_difficulty": "0",
        }

    processor_result = {}
    processor_result['filtered_schema'] = result["filtered_schema"]
    processor_result['new_schema'] = result['new_schema']
    processor_result['augmented_explanation'] = result['augmented_explanation']
    processor_result['query_difficulty'] = result['query_difficulty']

    logging.info("Finish processor")

    return processor_result


def process(db_id: str, query: str, db_schema: str, model_name: str) -> dict:
    prompt = processor_template.format(db_id=db_id,
                                       query=query,
                                       db_schema=db_schema)

    reply = gemini_api_call_with_config(model_name=model_name, prompt=prompt)
    result = parse_response(reply)
    return result
