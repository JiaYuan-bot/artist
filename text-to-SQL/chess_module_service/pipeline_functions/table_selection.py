import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, llm_chain
from .utils import node_decorator, get_last_node_result, add_columns_to_tentative_schema
from runner.database_manager import DatabaseManager
from llm.utils import get_prompt_engine_parser


@node_decorator(check_schema_status=True)
def table_selection(
    task: Any,
    tentative_schema: Dict[str, List[str]],
    execution_history: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Selects tables based on the specified mode and updates the tentative schema.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.
    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema and selected tables.
    """
    logging.info("Starting table selection")
    mode = config["table_selection"]["mode"]
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
            node_name="table_selection",
            llm_config=llm_config,
            schema_string=schema_string)

        request_kwargs = {
            "HINT": task.get("evidence", ""),
            "QUESTION": task.get("question", ""),
        }

        logging.info("Initiating LLM chain call for table selection")

        output = llm_chain(prompt=prompt,
                           engine=engine,
                           parser=parser,
                           request_kwargs=request_kwargs)
        aggregated_result = aggregate_tables([output])
        table_names = aggregated_result["table_names"]
        print(table_names)

        ######################## cascade ########################
        # from concurrent.futures import ThreadPoolExecutor

        # k = 10

        # def run_llm_and_aggregate(prompt, engine, parser, request_kwargs):
        #     """Run LLM chain and aggregate the result"""
        #     output = llm_chain(prompt=prompt,
        #                        engine=engine,
        #                        parser=parser,
        #                        request_kwargs=request_kwargs)
        #     aggregated_result = aggregate_tables([output])
        #     table_names = aggregated_result["table_names"]
        #     return table_names

        # # Multi-threaded execution
        # with ThreadPoolExecutor(max_workers=k) as executor:
        #     futures = [
        #         executor.submit(run_llm_and_aggregate, prompt, engine, parser,
        #                         request_kwargs) for _ in range(k)
        #     ]

        #     cascade_outputs = [future.result() for future in futures]

        # from collections import Counter
        # cascade_outputs_tuples = [tuple(output) for output in cascade_outputs]
        # # print(cascade_outputs_tuples)
        # # Count the frequency of each output
        # counter = Counter(cascade_outputs_tuples)

        # # Get the most common output and its count
        # most_common_output, most_common_count = counter.most_common(1)[0]
        # table_names = list(most_common_output)

        # # Calculate the proportion
        # total_outputs = len(cascade_outputs)
        # proportion = 1.0 * most_common_count / total_outputs
        # print(proportion)
        # print(table_names)
        # cascade = 0
        # if proportion >= 0.9:
        #     print('pass')
        # else:
        #     cascade = 1
        #     llm_config["model"] = "gemini-2.5-pro"
        #     prompt, engine, parser = get_prompt_engine_parser(
        #         node_name="table_selection",
        #         llm_config=llm_config,
        #         schema_string=schema_string)
        #     output = llm_chain(prompt=prompt,
        #                        engine=engine,
        #                        parser=parser,
        #                        request_kwargs=request_kwargs)
        #     aggregated_result = aggregate_tables([output])
        #     table_names = aggregated_result["table_names"]
        ######################## cascade ########################

        result = {
            "chain_of_thought_reasoning":
            aggregated_result["chain_of_thought_reasoning"],
            "selected_tables":
            table_names,
        }
    elif mode == "corrects":
        logging.info("Retrieving correct tables from SQL task")
        table_names = DatabaseManager(db_id=task.get("db_id")).get_sql_tables(
            task.sql)
        result = {
            "selected_tables": table_names,
        }
    else:
        logging.error(f"Unknown mode for table selection: {mode}")
        raise ValueError(f"Unknown mode for table selection: {mode}")

    tentative_schema = {
        table_name: tentative_schema.get(table_name, [])
        for table_name in table_names
    }

    similar_columns = get_last_node_result(
        execution_history,
        "entity_retrieval")["similar_columns"] if get_last_node_result(
            execution_history, "entity_retrieval") else {}
    add_columns_to_tentative_schema(tentative_schema, similar_columns)
    tentative_schema = DatabaseManager(db_id=task.get(
        "db_id")).add_connections_to_tentative_schema(tentative_schema)

    result = {
        "tentative_schema": tentative_schema,
        # "cascade": cascade,
        **result
    }
    logging.info("Table selection completed successfully")
    return result


def aggregate_tables(
        tables_dicts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Aggregates tables from multiple responses and consolidates reasoning.

    Args:
        tables_dicts (List[Dict[str, Any]]): List of dictionaries containing table names and reasoning.

    Returns:
        Dict[str, List[str]]: Aggregated result with unique table names and consolidated reasoning.
    """
    logging.info("Aggregating tables from multiple responses")
    tables = []
    chain_of_thoughts = []
    for table_dict in tables_dicts:
        chain_of_thoughts.append(
            table_dict.get("chain_of_thought_reasoning", ""))
        response_tables = table_dict.get("table_names", [])
        for table in response_tables:
            if table.lower() not in [t.lower() for t in tables]:
                tables.append(table)

    aggregated_chain_of_thoughts = "\n----\n".join(chain_of_thoughts)
    aggregation_result = {
        "table_names": tables,
        "chain_of_thought_reasoning": aggregated_chain_of_thoughts,
    }
    # logging.info(f"Aggregated tables: {tables}")
    return aggregation_result


if __name__ == "__main__":
    result = {
        'chain_of_thought_reasoning':
        "The question asks for the lowest three eligible free rates for students aged 5-17 in continuation schools. The hint provides the formula for eligible free rates: `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`. This information is directly available in the `frpm` table. Additionally, the question specifies 'continuation schools'. The `frpm` table has a `School Type` column, but it's not explicitly stated if 'Continuation School' is a value in that column. However, the `schools` table has an `EdOpsName` column with an example value of 'Continuation School', which is a strong indicator that this table can be used to filter for continuation schools. The `frpm` table has a `CDSCode` which is a foreign key referencing `schools.CDSCode`. Therefore, to filter for continuation schools and retrieve the free meal counts and enrollment for ages 5-17, we need to join `frpm` and `schools` tables.",
        'table_names': ['frpm', 'schools']
    }
    aggregate_tables(result)
