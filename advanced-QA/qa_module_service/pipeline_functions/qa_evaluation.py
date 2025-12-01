import logging
from typing import Dict, List, Any
import logging
from attr import dataclass
import re
import string
import unicodedata

from .utils import node_decorator, get_last_node_result


@node_decorator()
def qa_evaluation(
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
    logging.info("Starting qa_evaluation")

    ground_truth = task.get("ground_truth")

    answer = get_last_node_result(execution_history).get("answer", "")
    f1_score = f1_score_str(answer, ground_truth)

    evaluation_result = {}
    evaluation_result = {"f1_score": f1_score, "answer": answer}

    logging.info("Finish qa_evaluation")
    return evaluation_result


def f1_score_str(pred, label):
    """F1 score for two strings

    Will first tokenize the strings by space and calculate F1 score
    """
    label = set(normalize_text(label).split())
    pred = set(normalize_text(pred).split())
    return f1_score_set(pred, label)


def f1_score_set(pred, label):
    # Calculate true positives, false positives, and false negatives
    true_positives = len(label & pred)
    false_positives = len(pred - label)
    false_negatives = len(label - pred)

    if true_positives == 0:
        return 0

    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def normalize_text(s):
    # Normalize Unicode characters
    s = unicodedata.normalize("NFD", s)
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles (a, an, the)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Fix extra whitespaces
    s = " ".join(s.split())
    return s
