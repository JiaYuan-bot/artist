import logging
from typing import Any, Dict, List

from llm.models import async_llm_chain_call
from runner.database_manager import DatabaseManager
from .utils import node_decorator, get_last_node_result, add_columns_to_tentative_schema

# Import generated protobuf modules
from llm.utils import get_prompt_engine_parser


@node_decorator(check_schema_status=False)
def column_filtering(
    task: Any,
    tentative_schema: Dict[str, Any],
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Filters columns based on profiles and updates the tentative schema.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema.
    """
    logging.info("Starting column filtering")

    schema_with_examples = get_last_node_result(
        execution_history,
        "entity_retrieval")["similar_values"] if get_last_node_result(
            execution_history, "entity_retrieval") else {}
    column_profiles = DatabaseManager(
        db_id=task.get("db_id")).get_column_profiles(
            schema_with_examples=schema_with_examples,
            use_value_description=True,
            with_keys=True,
            with_references=True)

    list_of_kwargs = []
    for table_name, columns in column_profiles.items():
        for column_name, column_profile in columns.items():
            kwargs = {
                "QUESTION": task.get("question", ""),
                "HINT": task.get("evidence", ""),
                "COLUMN_PROFILE": column_profile,
            }
            list_of_kwargs.append(kwargs)

    llm_config = config.get("llm_config", {})
    prompt, engine, parser = get_prompt_engine_parser(
        node_name="column_filtering", llm_config=llm_config)

    logging.info("Initiating asynchronous LLM chain call for column filtering")
    ######################## none cascade ########################
    response = async_llm_chain_call(prompt=prompt,
                                    engine=engine,
                                    parser=parser,
                                    request_list=list_of_kwargs,
                                    step="column_filtering",
                                    sampling_count=1)

    choices = []
    for col_choice in response:
        choices.append(col_choice[0]["is_column_information_relevant"].lower())
    print(choices)
    ######################## none cascade ########################

    ######################## cascade ########################
    # k = 2
    # cascade_outputs = []
    # for i in range(k):
    #     response = async_llm_chain_call(prompt=prompt,
    #                                     engine=engine,
    #                                     parser=parser,
    #                                     request_list=list_of_kwargs,
    #                                     step="column_filtering",
    #                                     sampling_count=1)
    #     choices = []
    #     for col_choice in response:
    #         choices.append(
    #             col_choice[0]["is_column_information_relevant"].lower())
    #     # print(choices)
    #     cascade_outputs.append(choices)

    # from collections import Counter
    # cascade_outputs_tuples = [tuple(output) for output in cascade_outputs]
    # # print(cascade_outputs_tuples)
    # # Count the frequency of each output
    # counter = Counter(cascade_outputs_tuples)

    # # Get the most common output and its count
    # most_common_output, most_common_count = counter.most_common(1)[0]
    # choices = list(most_common_output)

    # # Calculate the proportion
    # total_outputs = len(cascade_outputs)
    # proportion = 1.0 * most_common_count / total_outputs
    # print(proportion)
    # print(choices)
    # if proportion >= 1.0:
    #     print('pass')
    # else:
    #     llm_config["model"] = "gemini-2.5-pro"
    #     prompt, engine, parser = get_prompt_engine_parser(
    #         node_name="column_filtering", llm_config=llm_config)

    #     response = async_llm_chain_call(prompt=prompt,
    #                                     engine=engine,
    #                                     parser=parser,
    #                                     request_list=list_of_kwargs,
    #                                     step="column_filtering",
    #                                     sampling_count=1)
    #     choices = []
    #     for col_choice in response:
    #         choices.append(
    #             col_choice[0]["is_column_information_relevant"].lower())
    ######################## cascade ########################

    index = 0
    tentative_schema = {}
    for table_name, columns in column_profiles.items():
        tentative_schema[table_name] = []
        for column_name, column_profile in columns.items():
            try:
                chosen = choices[index]
                # chosen = (response[index][0]
                #           ["is_column_information_relevant"].lower() == "yes")
                if chosen:
                    tentative_schema[table_name].append(column_name)
            except Exception as e:
                logging.error(
                    f"Error in column filtering for table '{table_name}', column '{column_name}': {e}"
                )
            index += 1

    # similar_columns = get_last_node_result(
    #     execution_history,
    #     "entity_retrieval")["similar_columns"] if get_last_node_result(
    #         execution_history, "entity_retrieval") else DatabaseManager(
    #             db_id=task.get("db_id")).get_db_schema()
    similar_columns = get_last_node_result(
        execution_history,
        "entity_retrieval")["similar_columns"] if get_last_node_result(
            execution_history, "entity_retrieval") else {}
    add_columns_to_tentative_schema(tentative_schema, similar_columns)
    tentative_schema = DatabaseManager(db_id=task.get(
        "db_id")).add_connections_to_tentative_schema(tentative_schema)

    result = {"tentative_schema": tentative_schema, "cascade": 1}
    logging.info("Column filtering completed successfully")
    return result
