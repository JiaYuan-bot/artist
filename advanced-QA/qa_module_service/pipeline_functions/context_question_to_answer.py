import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path

from .utils import node_decorator, get_last_node_result, parse_answer_from_string

from .const import *
from llm.models import gemini_api_call_with_config
from dspy.dsp.utils import deduplicate


@node_decorator()
def context_question_to_answer(
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
    logging.info("Starting context_question_to_answer")

    question = task.get("question")
    context = get_last_node_result(execution_history).get("context")
    print(context)

    llm_config = config.get("llm_config", {})
    model_name = llm_config.get("model", "gemini-2.5-pro")
    template_name = llm_config.get("template_name",
                                   "context_question_to_answer")
    prompt = ''
    if template_name == "context_question_to_answer":
        prompt = context_question_to_answer_template.format(context=context,
                                                            question=question)
    else:
        prompt = context_question_to_answer_cot_template.format(
            context=context, question=question)

    reply = gemini_api_call_with_config(model_name=model_name, prompt=prompt)

    answer = parse_answer_from_string(reply)
    print(answer)

    result = {'answer': answer}

    logging.info("Finish context_question_to_answer")

    return result
