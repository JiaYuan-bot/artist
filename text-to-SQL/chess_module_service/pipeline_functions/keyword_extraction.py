import logging
from typing import Dict, Any

from llm.utils import get_prompt_engine_parser
from llm.models import async_llm_chain_call, llm_chain

from .utils import node_decorator


@node_decorator(check_schema_status=False)
def keyword_extraction(
    task: Any,
    tentative_schema: Dict[str, Any],
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retrieves context information based on the task's question and evidence.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.

    Returns:
        Dict[str, Any]: A dictionary containing the schema with descriptions.
    """
    logging.info("Starting keyword extraction")
    # print(task)
    request_kwargs = {
        "HINT": task.get("evidence", ""),
        "QUESTION": task.get("question", ""),
    }
    llm_config = config.get("llm_config", {})

    prompt, engine, parser = get_prompt_engine_parser(
        node_name="keyword_extraction", llm_config=llm_config)

    logging.info("Initiating LLM chain call for keyword extraction")

    ######################## none cascade########################
    output = llm_chain(prompt=prompt,
                       engine=engine,
                       parser=parser,
                       request_kwargs=request_kwargs)
    ######################## none cascade ########################

    ######################## cascade ########################
    # from concurrent.futures import ThreadPoolExecutor, as_completed
    # k = 10
    # cascade_outputs = []

    # def run_llm_chain(prompt, engine, parser, request_kwargs):
    #     """Wrapper function to run a single LLM chain call"""
    #     return llm_chain(prompt=prompt,
    #                      engine=engine,
    #                      parser=parser,
    #                      request_kwargs=request_kwargs)

    # # Use ThreadPoolExecutor for multi-threading
    # with ThreadPoolExecutor(max_workers=k) as executor:
    #     # Submit all tasks
    #     futures = [
    #         executor.submit(run_llm_chain, prompt, engine, parser,
    #                         request_kwargs) for _ in range(k)
    #     ]

    #     # Collect results as they complete
    #     for future in as_completed(futures):
    #         output = future.result()
    #         cascade_outputs.append(output)

    # from collections import Counter
    # cascade_outputs_tuples = [tuple(output) for output in cascade_outputs]
    # # print(cascade_outputs_tuples)
    # # Count the frequency of each output
    # counter = Counter(cascade_outputs_tuples)

    # # Get the most common output and its count
    # most_common_output, most_common_count = counter.most_common(1)[0]
    # output = list(most_common_output)

    # # Calculate the proportion
    # total_outputs = len(cascade_outputs)
    # proportion = 1.0 * most_common_count / total_outputs
    # print(proportion)
    # cascade = 0
    # if proportion >= 0.9:
    #     print('pass')
    # else:
    #     cascade = 1
    #     llm_config["model"] = "gemini-2.5-pro"
    #     prompt, engine, parser = get_prompt_engine_parser(
    #         node_name="keyword_extraction", llm_config=llm_config)
    #     output = llm_chain(prompt=prompt,
    #                        engine=engine,
    #                        parser=parser,
    #                        request_kwargs=request_kwargs)

    # result = {"keywords": output, "cascade": cascade}
    ######################## cascade ########################

    result = {"keywords": output}
    logging.info("Finishing keyword extraction")

    # logging.info(f"Keywords extracted: {output}")
    return result
