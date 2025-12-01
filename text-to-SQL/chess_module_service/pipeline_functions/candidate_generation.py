import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, llm_chain
from runner.database_manager import DatabaseManager
from .utils import node_decorator, get_last_node_result

# Import generated protobuf modules
from llm.utils import get_prompt_engine_parser


@node_decorator(check_schema_status=False)
def candidate_generation(
    task: Any,
    tentative_schema: Dict[str, Any],
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generates candidate SQL queries based on the task's question and evidence.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.

    Returns:
        Dict[str, Any]: A dictionary containing the best SQL query result.
    """
    logging.info("Starting candidate generation")

    schema_with_examples = get_last_node_result(
        execution_history, node_type="entity_retrieval"
    )["similar_values"] if get_last_node_result(
        execution_history, node_type="entity_retrieval") else {}
    schema_with_descriptions = get_last_node_result(
        execution_history, node_type="context_retrieval"
    )["schema_with_descriptions"] if get_last_node_result(
        execution_history, node_type="context_retrieval") else {}

    schema_string = DatabaseManager(
        db_id=task.get("db_id")).get_database_schema_string(
            tentative_schema,
            schema_with_examples,
            schema_with_descriptions,
            include_value_description=True)
    print(schema_string)

    llm_config = config.get("llm_config", {})
    prompt, engine, parser = get_prompt_engine_parser(
        node_name="candidate_generation",
        llm_config=llm_config,
        schema_string=schema_string)

    # print(prompt)

    request_kwargs = {
        "HINT": task.get("evidence", ""),
        "QUESTION": task.get("question", ""),
    }

    logging.info("Initiating LLM chain call for candidate generation")
    try:
        output = llm_chain(prompt=prompt,
                           engine=engine,
                           parser=parser,
                           request_kwargs=request_kwargs)
    except Exception as e:
        logging.error("candidate_genaration raise error")
        raise e
    print(output['SQL'])

    ####################### cascade ########################
    # from concurrent.futures import ThreadPoolExecutor

    # k = 10

    # def run_llm_and_extract_sql(prompt, engine, parser, request_kwargs):
    #     """Run LLM chain and extract SQL"""
    #     output = llm_chain(prompt=prompt,
    #                        engine=engine,
    #                        parser=parser,
    #                        request_kwargs=request_kwargs)
    #     sql = output["SQL"]
    #     return sql

    # # Multi-threaded execution
    # with ThreadPoolExecutor(max_workers=k) as executor:
    #     futures = [
    #         executor.submit(run_llm_and_extract_sql, prompt, engine, parser,
    #                         request_kwargs) for _ in range(k)
    #     ]

    #     cascade_outputs = [future.result() for future in futures]

    # from collections import Counter
    # counter = Counter(cascade_outputs)
    # # Get the most common output and its count
    # most_common_output, most_common_count = counter.most_common(1)[0]
    # output = {'SQL': most_common_output}

    # # Calculate the proportion
    # total_outputs = len(cascade_outputs)
    # proportion = 1.0 * most_common_count / total_outputs
    # print(proportion)
    # print(output)
    # cascade = 0
    # if proportion >= 0.9:
    #     print('pass')
    # else:
    #     cascade = 1
    #     llm_config["model"] = "gemini-2.5-pro"
    #     prompt, engine, parser = get_prompt_engine_parser(
    #         node_name="candidate_generation",
    #         llm_config=llm_config,
    #         schema_string=schema_string)
    #     output = llm_chain(prompt=prompt,
    #                        engine=engine,
    #                        parser=parser,
    #                        request_kwargs=request_kwargs)

    ####################### cascade ########################
    # output['cascade'] = cascade
    logging.info("Candidate generation completed successfully")
    return output
