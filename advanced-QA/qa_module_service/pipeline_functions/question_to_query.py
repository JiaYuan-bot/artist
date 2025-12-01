import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path

from .utils import node_decorator, get_last_node_result, search_wikipedia, parse_query_from_string

from .const import *
from llm.models import gemini_api_call_with_config


@node_decorator()
def question_to_query(
    task: Any,
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
):
    """
    :param task:{
                    "id": 
                    "question": question text,
                    "ground_truth": ground truth text,
                }
    :return: 
    """
    logging.info("Starting question_to_query")

    question = task.get('question')

    llm_config = config.get("llm_config", {})
    model_name = llm_config.get("model", "gemini-2.5-pro")

    prompt = question_to_query_template.format(question=question)
    reply = gemini_api_call_with_config(model_name=model_name, prompt=prompt)

    query = parse_query_from_string(reply)
    print(query)

    result = {'query_1': query}
    logging.info("Finish question_to_query")

    return result
