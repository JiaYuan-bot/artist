import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, llm_chain
from runner.database_manager import DatabaseManager
from .utils import node_decorator, get_last_node_result

# Import generated protobuf modules
from llm.utils import get_prompt_engine_parser


@node_decorator(check_schema_status=True)
def column_selection(
    task: Any,
    tentative_schema: Dict[str, Any],
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Selects columns based on the specified mode and updates the tentative schema.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema and selected columns.
    """
    logging.info("Starting column selection")
    mode = config["column_selection"]["mode"]
    llm_config = config.get("llm_config", {})

    if mode == "ask_model":
        schema_with_examples = get_last_node_result(
            execution_history,
            "entity_retrieval")["similar_values"] if get_last_node_result(
                execution_history, "entity_retrieval") else {}
        schema_with_descriptions = get_last_node_result(
            execution_history, "context_retrieval"
        )["schema_with_descriptions"] if get_last_node_result(
            execution_history, "context_retrieval") else {}

        schema_string = DatabaseManager(
            db_id=task.get("db_id")).get_database_schema_string(
                tentative_schema,
                schema_with_examples,
                schema_with_descriptions,
                include_value_description=True)

        prompt, engine, parser = get_prompt_engine_parser(
            node_name="column_selection",
            llm_config=llm_config,
            schema_string=schema_string)

        request_kwargs = {
            "HINT": task.get("evidence", ""),
            "QUESTION": task.get("question", ""),
        }

        logging.info("Initiating LLM chain call for column selection")
        output = llm_chain(prompt=prompt,
                           engine=engine,
                           parser=parser,
                           request_kwargs=request_kwargs)

        aggregated_result = aggregate_columns([output],
                                              list(tentative_schema.keys()))
        column_names = aggregated_result
        print(column_names)

        ######################## cascade ########################
        # from concurrent.futures import ThreadPoolExecutor
        # k = 10

        # def run_llm_and_aggregate(prompt, engine, parser, request_kwargs,
        #                           schema_keys):
        #     """Run LLM chain and aggregate columns"""
        #     output = llm_chain(prompt=prompt,
        #                        engine=engine,
        #                        parser=parser,
        #                        request_kwargs=request_kwargs)
        #     aggregated_result = aggregate_columns([output], schema_keys)
        #     return aggregated_result

        # # Get schema keys once before threading
        # schema_keys = list(tentative_schema.keys())

        # # Multi-threaded execution
        # with ThreadPoolExecutor(max_workers=k) as executor:
        #     futures = [
        #         executor.submit(run_llm_and_aggregate, prompt, engine, parser,
        #                         request_kwargs, schema_keys) for _ in range(k)
        #     ]

        #     cascade_outputs = [future.result() for future in futures]

        # from collections import Counter
        # import json
        # # Convert each dict to a JSON string (hashable)
        # cascade_outputs_strings = [
        #     json.dumps(output, sort_keys=True) for output in cascade_outputs
        # ]
        # # Count frequencies
        # counter = Counter(cascade_outputs_strings)

        # # Get the most common output and its count
        # most_common_output, most_common_count = counter.most_common(1)[0]
        # column_names = json.loads(most_common_output)

        # # Calculate the proportion
        # total_outputs = len(cascade_outputs)
        # proportion = 1.0 * most_common_count / total_outputs
        # print(proportion)
        # print(column_names)
        # cascade = 0
        # if proportion >= 0.9:
        #     print('pass')
        # else:
        #     cascade = 1
        #     llm_config["model"] = "gemini-2.5-pro"
        #     prompt, engine, parser = get_prompt_engine_parser(
        #         node_name="column_selection",
        #         llm_config=llm_config,
        #         schema_string=schema_string)
        #     output = llm_chain(prompt=prompt,
        #                        engine=engine,
        #                        parser=parser,
        #                        request_kwargs=request_kwargs)
        #     column_names = aggregate_columns([output],
        #                                      list(tentative_schema.keys()))
        ######################## cascade ########################

        # chain_of_thought_reasoning = aggregated_result.pop(
        #     "chain_of_thought_reasoning")
        result = {
            "tentative_schema": column_names,
            "model_selected_columns": column_names,
            # "cascade": cascade
            # "chain_of_thought_reasoning": chain_of_thought_reasoning
        }
    elif mode == "corrects":
        logging.info("Retrieving correct columns from SQL task")
        column_names = DatabaseManager(
            db_id=task.get("db_id")).get_sql_columns_dict(task.sql)
        result = {
            "tentative_schema": column_names,
            "selected_columns": column_names
        }
    else:
        logging.error(f"Unknown mode for column selection: {mode}")
        raise ValueError(f"Unknown mode for column selection: {mode}")

    logging.info("Column selection completed successfully")
    return result


def aggregate_columns(columns_dicts: List[Dict[str, Any]],
                      selected_tables: List[str]) -> Dict[str, List[str]]:
    """
    Aggregates columns from multiple responses and consolidates reasoning.

    Args:
        columns_dicts (List[Dict[str, Any]]): List of dictionaries containing column names and reasoning.
        selected_tables (List[str]): List of selected tables.

    Returns:
        Dict[str, List[str]]: Aggregated result with unique column names and consolidated reasoning.
    """
    # logging.info("Aggregating columns from multiple responses")
    columns = {}
    chain_of_thoughts = []
    for column_dict in columns_dicts:
        valid_column_dict = False
        for key, value in column_dict.items():
            if key == "chain_of_thought_reasoning":
                dict_cot = value
            else:  # key is table name
                table_name = key
                if table_name.startswith("`"):
                    table_name = table_name[1:-1]
                column_names = value
                if table_name.lower() in [t.lower() for t in selected_tables]:
                    for column_name in column_names:
                        if column_name.startswith("`"):
                            column_name = column_name[1:-1]
                        if table_name not in columns:
                            columns[table_name] = []
                        if column_name.lower() not in [
                                col.lower() for col in columns[table_name]
                        ]:
                            columns[table_name].append(column_name)
                        valid_column_dict = True
        if valid_column_dict:
            chain_of_thoughts.append(dict_cot)

    aggregated_chain_of_thoughts = "\n----\n".join(chain_of_thoughts)
    aggregation_result = columns
    # aggregation_result[
    #     "chain_of_thought_reasoning"] = aggregated_chain_of_thoughts

    # logging.info(f"Aggregated columns: {columns}")
    return aggregation_result
