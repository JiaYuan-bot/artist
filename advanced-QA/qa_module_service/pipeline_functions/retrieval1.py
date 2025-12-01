import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path

from .utils import node_decorator, get_last_node_result, search_wikipedia, parse_query_from_string

from .const import *
from llm.models import gemini_api_call_with_config
from .multihop_rag import retrieve_chuncks


@node_decorator()
def retrieval1(
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
    logging.info("Starting retrieval1")

    question = task.get('question')
    top_k = config["retrieval1"].get("top_k", 3)
    top_k=3

    last_node_result = get_last_node_result(execution_history)

    if last_node_result.get('node_type', "") == 'question_to_query':
        query = last_node_result["query_1"]
        if query == "":
            query = question
        context = retrieve_chuncks(query, top_k=top_k)
    else:
        context = retrieve_chuncks(question, top_k=top_k)

    print(context)

    result = {'context': context}

    logging.info("Finish retrieval1")

    return result
