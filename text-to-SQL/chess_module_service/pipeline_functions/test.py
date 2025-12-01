import logging
from typing import Dict, List, Any


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
    logging.info(f"Aggregated tables: {tables}")
    return aggregation_result


if __name__ == "__main__":
    result = {
        'chain_of_thought_reasoning':
        "The question asks for the lowest three eligible free rates for students aged 5-17 in continuation schools. The hint provides the formula for eligible free rates: `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`. This information is directly available in the `frpm` table. Additionally, the question specifies 'continuation schools'. The `frpm` table has a `School Type` column, but it's not explicitly stated if 'Continuation School' is a value in that column. However, the `schools` table has an `EdOpsName` column with an example value of 'Continuation School', which is a strong indicator that this table can be used to filter for continuation schools. The `frpm` table has a `CDSCode` which is a foreign key referencing `schools.CDSCode`. Therefore, to filter for continuation schools and retrieve the free meal counts and enrollment for ages 5-17, we need to join `frpm` and `schools` tables.",
        'table_names': ['frpm', 'schools']
    }
    aggregate_tables(result)
