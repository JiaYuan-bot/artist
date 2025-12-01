import os
import json
import sqlite3

import re
import json

llm_1 = {
    'gemini-2.5-pro': '_large',
    'gemini-2.5-flash': '_large',
    'gemini-2.5-flash-lite': '_small',
}

llm_2 = {
    'gemini-2.5-pro': '_large',
    'gemini-2.5-flash': '_medium',
    'gemini-2.5-flash-lite': '_small',
}

default_routing_path = [
    'preprocess', 'processor_large', 'composer_large', 'translator',
    'validator_large', 'nv_evaluation', 'END'
]


def build_routing_from_configuration(
    path:
    str = '/home/jiayuan/nl2sql/qa_service_actor/result/test_set_llm/configurations_2.json'
):
    with open(path, 'r', encoding='utf-8') as f:
        configurations = json.load(f)

    all_routing_paths = []
    for idx, configuration in enumerate(configurations):
        routing_path = []
        for key in configuration['configuration'].keys():
            if key == 'question_to_query' or key == 'rerank':
                routing_path.append(
                    key + llm_1[configuration['configuration'][key]
                                ['configs'].get('LLM', 'gemini-2.5-pro')])
            elif key == 'retrieval1':
                routing_path.append('retrieval1_3')
            elif key == 'retrieval2':
                routing_path.append(
                    'context_question_to_query' +
                    llm_1[configuration['configuration'][key]['configs'].get(
                        'LLM', 'gemini-2.5-pro')])
                routing_path.append('retrieval2_3')
            elif key == 'context_question_to_answer':
                routing_path.append(
                    key +
                    llm_2[configuration['configuration'][key]['configs'].get(
                        'LLM', 'gemini-2.5-pro')] + '_cot')
            elif key == 'qa_evaluation':
                routing_path.append('qa_evaluation')
            else:
                print(f'id:{idx}, error module: {key}')
                routing_path = []
                break
        routing_path.append('END')
        print(routing_path)
        if routing_path:
            all_routing_paths.append(routing_path)
        else:
            all_routing_paths.append(default_routing_path)
    print(len(all_routing_paths))
    return all_routing_paths


if __name__ == "__main__":
    build_routing_from_configuration("configurations_2.json")
