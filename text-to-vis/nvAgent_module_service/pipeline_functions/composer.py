import logging
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import seaborn as sns
import sqlglot
import re
import time
import pandas as pd
from pathlib import Path
from tqdm import trange

from .utils import node_decorator, get_node_result, get_last_node_result
from .utils import parse_vql_from_string
from .const import *
from llm.models import gemini_api_call_with_config
"""
Decompose the question and solve them using CoT
"""


@node_decorator()
def composer(
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
    :return: VQL 
    """
    logging.info("Starting composer")

    prompt = ''
    query = task.get('query')
    last_node_result = get_last_node_result(execution_history)

    ### without processor ###
    if last_node_result['node_type'] == 'preprocess':
        schema_info = last_node_result.get('old_schema')
        prompt = single_template.format(query=query, desc_str=schema_info)

    ### with processor ###
    if last_node_result['node_type'] == 'processor':
        schema_info, augmented_explanation = last_node_result.get(
            'new_schema'), last_node_result.get('augmented_explanation')
        query_difficulty = last_node_result.get('query_difficulty', "0")

        if query_difficulty == "SINGLE":
            prompt_template = single_template
            schema_info = get_node_result(execution_history,
                                          'preprocess').get('old_schema')
            prompt = prompt_template.format(query=query, desc_str=schema_info)
        else:
            prompt_template = multiple_template
            basic_prompt_template = basic_composer_template
            prompt = prompt_template.format(
                query=query,
                desc_str=schema_info,
                augmented_explanation=augmented_explanation)

    # # without processor
    # query, schema_info = message.get('query'), message.get('old_schema')
    # prompt = single_template.format(query=query, desc_str=schema_info)

    # # without composer
    # prompt = basic_composer_template.format(
    #     query=query,
    #     desc_str=schema_info,
    #     augmented_explanation=augmented_explanation)

    llm_config = config.get("llm_config", {})
    model_name = llm_config.get("model", "gemini-2.5-pro")

    reply = gemini_api_call_with_config(model_name=model_name, prompt=prompt)

    vql = ''

    try:
        vql = parse_vql_from_string(reply)
        if vql == '':
            logging.error("No VQL found in the response")
    except Exception as e:
        res = f'error!: {str(e)}'
        logging.error(res)

    composer_result = {}
    composer_result['final_vql'] = vql

    logging.info("Finish composer")
    return composer_result
