import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Dict, Any, Optional, Protocol, Tuple
import re, json, math, hashlib

from .utils import node_decorator, get_last_node_result, search_wikipedia, parse_query_from_string

from .const import *
from llm.models import gemini_api_call_with_config

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_obj(s: str) -> Dict[str, Any]:
    m = _JSON_OBJ_RE.search(s)
    if not m:
        raise ValueError("No JSON object found.")
    return json.loads(m.group(0))


def _extract_json_arr(s: str) -> List[Dict[str, Any]]:
    m = _JSON_ARR_RE.search(s)
    if not m:
        raise ValueError("No JSON array found.")
    return json.loads(m.group(0))


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    id: str
    text: str


@dataclass
class ScoredChunk:
    id: str
    text: str
    score: float
    justification: str


def score_pairwise(query: str, chunk: Chunk, model_name: str) -> ScoredChunk:
    prompt = PAIRWISE_RERANK_PROMPT.format(query=_norm_text(query),
                                           passage=_norm_text(chunk.text))

    reply = gemini_api_call_with_config(model_name=model_name, prompt=prompt)
    try:
        obj = _extract_json_obj(reply)
        score = float(int(obj.get("score", 0)))
        justification = str(obj.get("justification", "")).strip()
    except Exception:
        score = 0.0
        justification = "Failed to parse JSON; treated as 0."

    blended = score

    return ScoredChunk(
        id=chunk.id,
        text=chunk.text,
        score=float(blended),
        justification=justification,
    )


@node_decorator()
def rerank(task: Any, execution_history: Dict[str, Any],
           config: Dict[str, Any]) -> List[ScoredChunk]:
    """
    :param task:{
                    "id": 
                    "question": question text,
                    "ground_truth": ground truth text,
                }
    :return: 
    """
    logging.info("Starting rerank")

    query = task.get("question")
    context = get_last_node_result(execution_history).get('context')
    chunks: list[Chunk] = [
        Chunk(id=str(i), text=c) for i, c in enumerate(context)
    ]

    llm_config = config.get("llm_config", {})
    model_name = llm_config.get("model", "gemini-2.5-pro")

    # Pairwise scoring
    scored = [score_pairwise(query, c, model_name) for c in chunks]
    scored.sort(key=lambda x: x.score, reverse=True)

    # set >0 (e.g., 8) to do a listwise second pass on the top-M
    listwise_refine_m = config["rerank"].get("listwise_refine_m", 0)

    if listwise_refine_m > 0 and len(scored) > 1:
        m = min(listwise_refine_m, len(scored))
        topm = scored[:m]
        candidates_block = "\n".join(
            f"- id: {c.id}\n  passage: { _norm_text(c.text) }" for c in topm)
        lprompt = LISTWISE_REFINE_PROMPT.format(
            query=_norm_text(query), candidates_block=candidates_block)

        reply = gemini_api_call_with_config(model_name=model_name,
                                            prompt=lprompt)

        try:
            arr = _extract_json_arr(reply)
            order: Dict[str, Tuple[int, float]] = {}
            # Higher score wins; preserve order by index as tie-breaker
            for idx, item in enumerate(arr):
                cid = str(item.get("id", ""))
                cscore = float(int(item.get("score", 0)))
                order[cid] = (idx, cscore)
            # Apply refined scores/ordering only to the top-M slice
            topm_sorted = sorted(
                topm,
                key=lambda c:
                (order.get(c.id, (9999, -1))[1], -order.get(c.id,
                                                            (9999, -1))[0]),
                reverse=True,
            )
            # Replace the first M with refined list; keep rest as-is
            scored = topm_sorted + scored[m:]
        except Exception:
            # If listwise parsing fails, keep pairwise order
            pass

    context_reranked = [sc.text for sc in scored]
    context_reranked = context_reranked[:6]
    result = {'context': context_reranked}
    # print(context_reranked)

    logging.info("Finish rerank")
    return result
