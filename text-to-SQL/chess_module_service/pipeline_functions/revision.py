import difflib
import logging
from typing import Dict, List, Tuple, Any

from llm.models import async_llm_chain_call, llm_chain
from database_utils.schema import DatabaseSchema
from runner.database_manager import DatabaseManager
from .utils import node_decorator, get_last_node_result

# Import generated protobuf modules
from llm.utils import get_prompt_engine_parser


@node_decorator(check_schema_status=False)
def revision(
    task: Any,
    tentative_schema: Dict[str, List[str]],
    execution_history: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Revises the predicted SQL query based on task evidence and schema information.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.
        config (Dict[str, Any]): Configuration dictionary containing LLM(tool...) settings.

    Returns:
        Dict[str, Any]: A dictionary containing the revised SQL query and reasoning.
    """
    logging.info("Starting SQL revision")

    schema_with_examples = get_last_node_result(
        execution_history, "entity_retrieval").get(
            "similar_values", {}) if get_last_node_result(
                execution_history, "entity_retrieval") else {}
    schema_with_descriptions = get_last_node_result(
        execution_history, "context_retrieval").get(
            "schema_with_descriptions", {}) if get_last_node_result(
                execution_history, "context_retrieval") else {}

    complete_schema = DatabaseManager(db_id=task.get("db_id")).get_db_schema()
    schema_string = DatabaseManager(
        db_id=task.get("db_id")).get_database_schema_string(
            complete_schema,
            schema_with_examples,
            schema_with_descriptions,
            include_value_description=True)

    llm_config = config.get("llm_config", {})
    prompt, engine, parser = get_prompt_engine_parser(
        node_name="revision",
        llm_config=llm_config,
        schema_string=schema_string)

    predicted_query = get_last_node_result(execution_history,
                                           "candidate_generation")["SQL"]

    try:
        query_result = DatabaseManager(
            db_id=task.get("db_id")).validate_sql_query(
                sql=predicted_query)['RESULT']
    except Exception as e:
        query_result = str(e)
        logging.error(f"Error validating SQL query: {e}")

    try:
        missing_entities = find_wrong_entities(
            predicted_query,
            schema_with_examples,
            db_id=task.get("db_id"),
        )
    except Exception as e:
        missing_entities = {}
        logging.error(f"Error finding wrong entities: {e}")

    request_kwargs = {
        "SQL": predicted_query,
        "QUESTION": task.get("question", ""),
        "MISSING_ENTITIES": missing_entities,
        "EVIDENCE": task.get("evidence", ""),
        "QUERY_RESULT": query_result,
    }

    logging.info("Initiating LLM chain call for SQL revision")
    try:
        output = llm_chain(prompt=prompt,
                           engine=engine,
                           parser=parser,
                           request_kwargs=request_kwargs)
    except Exception as e:
        logging.error("revision raise error")
        raise e

    ######################## cascade ########################
    # from concurrent.futures import ThreadPoolExecutor

    # k = 10

    # def run_llm_and_extract_sql(prompt, engine, parser, request_kwargs):
    #     """Run LLM chain and extract revised SQL"""
    #     output = llm_chain(prompt=prompt,
    #                        engine=engine,
    #                        parser=parser,
    #                        request_kwargs=request_kwargs)
    #     sql = output["revised_SQL"]
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
    # output = {'revised_SQL': most_common_output}

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
    #         node_name="revision",
    #         llm_config=llm_config,
    #         schema_string=schema_string)
    #     output = llm_chain(prompt=prompt,
    #                        engine=engine,
    #                        parser=parser,
    #                        request_kwargs=request_kwargs)
    ######################## cascade ########################

    print(output["revised_SQL"])
    # result = {"SQL": output["revised_SQL"], "cascade": cascade}
    result = {"SQL": output["revised_SQL"]}

    logging.info("SQL revision completed successfully")
    return result


def find_wrong_entities(sql: str,
                        similar_values: Dict[str, Dict[str, List[str]]],
                        db_id: str,
                        similarity_threshold: float = 0.4) -> str:
    """
    Finds and returns a string listing entities in the SQL that do not match the database schema.

    Args:
        sql (str): The SQL query to check.
        similar_values (Dict[str, Dict[str, List[str]]]): Dictionary of similar values for columns.
        similarity_threshold (float, optional): The similarity threshold for matching values. Defaults to 0.4.

    Returns:
        str: A string listing the mismatched entities and suggestions.
    """
    logging.info("Finding wrong entities in the SQL query")
    wrong_entities = ""
    used_entities = DatabaseManager(
        db_id=db_id).get_sql_condition_literals(sql)
    similar_values_database_schema = DatabaseSchema.from_schema_dict_with_examples(
        similar_values)

    for table_name, column_info in used_entities.items():
        for column_name, column_values in column_info.items():
            target_column_info = similar_values_database_schema.get_column_info(
                table_name, column_name)
            if not target_column_info:
                continue
            for value in column_values:
                column_similar_values = target_column_info.examples
                if value not in column_similar_values:
                    most_similar_entity, similarity = _find_most_syntactically_similar_value(
                        value, column_similar_values)
                    if similarity > similarity_threshold:
                        wrong_entities += f"Column {column_name} in table {table_name} does not contain the value '{value}'. The correct value is '{most_similar_entity}'.\n"

    for used_table_name, used_column_info in used_entities.items():
        for used_column_name, used_values in used_column_info.items():
            for used_value in used_values:
                for table_name, column_info in similar_values.items():
                    for column_name, column_values in column_info.items():
                        if (used_value in column_values) and (
                                column_name.lower()
                                != used_column_name.lower()):
                            wrong_entities += f"Value {used_value} that you used in the query appears in the column {column_name} of table {table_name}.\n"
    return wrong_entities


def _find_most_syntactically_similar_value(
        target_value: str, candidate_values: List[str]) -> Tuple[str, float]:
    """
    Finds the most syntactically similar value to the target value from the candidate values.

    Args:
        target_value (str): The target value to match.
        candidate_values (List[str]): The list of candidate values.

    Returns:
        Tuple[str, float]: The most similar value and the similarity score.
    """
    most_similar_entity = max(candidate_values,
                              key=lambda value: difflib.SequenceMatcher(
                                  None, value, target_value).ratio(),
                              default=None)
    max_similarity = difflib.SequenceMatcher(
        None, most_similar_entity,
        target_value).ratio() if most_similar_entity else 0
    return most_similar_entity, max_similarity
