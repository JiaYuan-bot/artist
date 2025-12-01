import os
import json
import sqlite3

import re
import json

llm = {
    'gemini-2.5-pro': '_large',
    'gemini-2.5-flash': '_small',
    'gemini-2.5-flash-lite': '_small',
}

llm_2 = {
    'gemini-2.5-pro': '_large',
    'gemini-2.5-flash': '_medium',
    'gemini-2.5-flash-lite': '_small',
}

default_routing_path = [
    'keyword_extraction_small', 'column_filter_small', 'table_selection_small',
    'column_selection_small', 'candidate_gen_large', 'revision_large',
    'evaluation', 'END'
]


def build_routing_from_configuration(
    path:
    str = '/home/jiayuan/nl2sql/chess_service_actor/result/test_set_llm/configurations.json'
):
    with open(path, 'r', encoding='utf-8') as f:
        configurations = json.load(f)

    all_routing_paths = []
    for idx, configuration in enumerate(configurations):
        routing_path = []
        routing_path.append('keyword_extraction_small')
        for key in configuration['configuration'].keys():
            if key == 'information_retrieval':
                routing_path.append('entity_retrieval_small')
                routing_path.append('context_retrieval_k5')
            elif key == 'schema_linking':
                routing_path.append(
                    'column_filter' +
                    llm[configuration['configuration'][key]['configs'].get(
                        'LLM', 'gemini-2.5-pro')])
                routing_path.append(
                    'table_selection' +
                    llm[configuration['configuration'][key]['configs'].get(
                        'LLM', 'gemini-2.5-pro')])
                routing_path.append(
                    'column_selection' +
                    llm[configuration['configuration'][key]['configs'].get(
                        'LLM', 'gemini-2.5-pro')])
            elif key == 'candidate_generation':
                routing_path.append(
                    'candidate_gen' +
                    llm_2[configuration['configuration'][key]['configs'].get(
                        'LLM', 'gemini-2.5-pro')])
            elif key == 'revison':
                routing_path.append('revision_large')
            elif key == 'evaluation':
                routing_path.append('evaluation')
            else:
                print(f'id:{idx}, error module: {key}')
                routing_path = default_routing_path
                break
        routing_path.append('END')
        if routing_path[-1] != 'END' or routing_path[-2] != 'evaluation':
            print(f'id:{idx}, error workflow')
            routing_path = default_routing_path
        # print(routing_path)
        if routing_path:
            all_routing_paths.append(routing_path)
        else:
            all_routing_paths.append(default_routing_path)
    print(len(all_routing_paths))
    return all_routing_paths


if __name__ == "__main__":
    build_routing_from_configuration("configurations.json")
