import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path

from .utils import node_decorator, get_node_result, get_last_node_result, search_wikipedia, parse_query_from_string

from .const import *
from llm.models import gemini_api_call_with_config
from dspy.dsp.utils import deduplicate
from .multihop_rag import retrieve_chuncks


@node_decorator()
def retrieval2(
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
    logging.info("Starting retrieval2")

    top_k = config["retrieval2"].get("top_k", 3)

    context = get_node_result(execution_history, "retrieval1").get('context')

    last_node_result = get_last_node_result(execution_history)

    query_2 = last_node_result["query_2"]

    context_new = retrieve_chuncks(query_2, top_k=top_k)
    context = deduplicate(context + context_new)
    # print(context)

    result = {'context': context}

    logging.info("Finish retrieval2")

    return result
