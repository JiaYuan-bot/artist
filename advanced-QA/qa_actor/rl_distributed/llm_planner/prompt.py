prompt = """
You are a Question-to-Answer(QA) Workflow Planner.
Your job is to plan (not execute) an end-to-end Question-to-Answer(QA) workflow for a single query.
Select which modules to run (you may skip modules) and choose their configurations to minimize latency while meeting a quality threshold.

I will give user question and detailed workflow modules meta information with their condidate configurations. 
You should think carefully, output a workflow that you think it is the best fit for the user query. The workflow should minimize latency while meeting a quality threshold.

Inputs:
1. User question

Workflow Modules(Each module has functional description and allowed configs. Also, we will give the skippable modules):


question_to_query:
  description: rewrite user's question into a RAG search query
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
  
retrieval1: 
  description: retrieve top_k most similar data chuncks form a vector database.
  configs: 
    top_k: [3, 5, 9]

retrieval2:
  description: rewrite user's question into a new RAG search query based on retrieval1's result.
  configs:  
    LLM: [gemini-2.5-pro, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
    top_k: [3, 5, 9]

rerank:
  description: rerank the retrieved chunks based on their relevance of the user's question
  configs:  
    LLM: [gemini-2.5-pro, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
  
context_question_to_answer: 
  description: generate answer based on user's questino and retrieved chuncks.
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
    prompt: [basic, cot]

qa_evaluation: 
  description: Score the generated answer
  configs:


All skippable cases:
[
  (question_to_query),
  (retrieval2),
  (rerank),
  (question_to_query, retrieval2),
  (question_to_query, rerank),
  (retrieval2, rerank),
  (question_to_query, retrieval2, rerank)
]

(question_to_query) means skipping only question_to_query
(question_to_query, retrieval2, rerank) means skip question_to_query, retrieval2, rerank


  
Rules and Heuristics

0. Carefully analysis the user question complexity, simpler indicate a light-weight workflow.
1. Prefer the cheapest workflow that plausibly generate correct answer.
2. Skip aggressively when safe (simple question). But when you think this query is complex, don't be too aggressive. Keep safe!
3. Prefer small/medium LLM; escalate only if risk of failing the threshold is high.
4. Just-in-time activation: decide knobs for a module only if you include it.
5. Respect permit skippable modules (e.g., cannot retrieval1, context_question_to_answer, qa_evaluation).
7. Don't generate answer here; plan the workflow only.
8. The calling order of modules cannot be disrupted

Pay attention. Rerank module has high latency. You should skip rerank aggressively. Only when you think it is necessary you could maintain it.
Pay attention. Retrieval2 is benificial a lot. And compared with rerank, it is important for complex queries which require multi-hop evidence retrieval.
Think carefully before you want to skip it.

Output format! return a single object matching this example schema, do not output json format!

question_to_query:
  configs: 
    LLM: gemini-2.5-flash
    temperature: 0.8
  
retrieval1: 
  configs: 
    top_k: 5

retrieval2:
  configs:  
    LLM: gemini-2.5-pro
    temperature: 0.8
    top_k: 3

rerank:
  configs:  
    LLM: gemini-2.5-pro
    temperature: 1.0
  
context_question_to_answer: 
  configs: 
    LLM: gemini-2.5-pro
    temperature: 0.8
    prompt: cot

qa_evaluation: 
  configs:


Example

User question: Between the report from Fortune published on October 4, 2023, and the report from TechCrunch, was there consistency in the portrayal of Sam Bankman-Fried's actions related to the allegations of fraud?

Planner output:

question_to_query:
  configs: 
    LLM: gemini-2.5-flash-lite
    temperature: 0.8
  
retrieval1: 
  configs: 
    top_k: 5
  
context_question_to_answer: 
  configs: 
    LLM: gemini-2.5-flash-lite
    temperature: 0.8
    prompt: cot

qa_evaluation: 
  configs:

---
Now I will give the user query, user hint, db schema. Please give me the workflow.
User question {question}


Planner output:
"""

import os
import json
import sqlite3

import re
import json
from typing import Dict
from functools import wraps
from typing import Dict, List, Any, Callable
import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from langchain_google_vertexai import VertexAI
from google.cloud import aiplatform
from typing import Dict, Any
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os
from pathlib import Path

GCP_PROJECT = "liquid-sylph-476522-r8"
GCP_REGION = "us-central1"

if GCP_PROJECT and GCP_REGION:
    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
    )
    vertexai.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
    )

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-2.5-flash-lite": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-flash-lite",
        }
    },
    "gemini-2.5-flash": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-flash",
        }
    },
    "gemini-2.5-pro": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-pro",
        }
    }
}


@dataclass
class QATask:
    """
    Represents a task with question and database details.

    Attributes:
        id (int): The unique integer identifier for the question.
        question (str): The question text.
        ground_truth (str): The ground truth answer.
        level (str): The difficulty level of the question.
    """
    # Changed type hint from str to int
    id: int = field(init=False)
    question: str = field(init=False)
    ground_truth: str = field(init=False)

    # level: str = field(init=False)

    # Updated __init__ to accept an integer ID
    def __init__(self, task_id: int, task_data: Dict[str, Any]):
        """
        Initializes a Task instance using an integer ID and data from a dictionary.

        Args:
            task_id (int): The sequential integer ID for the task.
            task_data (Dict[str, Any]): A dictionary containing the rest of the task data.
        """
        self.id = task_id
        self.question = task_data["query"]
        self.ground_truth = task_data["answer"]
        # self.level = task_data["level"]


def load_test_dataset(path: str) -> List[QATask]:
    """
    Load test dataset from JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of test samples
    """
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        tasks = [
            QATask(idx, task_data) for idx, task_data in enumerate(dataset)
        ]
    return tasks


def parse_workflow_to_json(workflow_str):
    """
    Parse workflow configuration string into JSON/dict format.
    
    Args:
        workflow_str: String containing workflow configuration
    
    Returns:
        dict: Parsed workflow configuration
    """
    result = {}
    current_module = None

    lines = workflow_str.strip().split('\n')

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Count leading spaces to determine indentation level
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        # Module name (no indentation, ends with colon)
        if indent == 0 and stripped.endswith(':'):
            current_module = stripped.rstrip(':')
            result[current_module] = {'configs': {}}

        # "configs:" line (skip it)
        elif stripped == 'configs:':
            continue

        # Config key-value pairs
        elif ':' in stripped and current_module:
            key, value = stripped.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Parse the value
            if value == '':
                parsed_value = None
            elif value.replace('.', '').replace('-', '').isdigit():
                # Number
                parsed_value = float(value) if '.' in value else int(value)
            else:
                # String
                parsed_value = value

            result[current_module]['configs'][key] = parsed_value

    return result


def gemini_api_call_with_config(model_name: str, prompt: str):
    if model_name not in ENGINE_CONFIGS:
        raise ValueError(f"Model {model_name} not found in ENGINE_CONFIGS")

    params = ENGINE_CONFIGS[model_name]["params"]

    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_REGION,
    )

    # Build generation config from your params
    generation_config = {
        "temperature": params.get("temperature", 0),
        "top_p": params.get("top_p", 0.5),
        "top_k": params.get("top_k", 3),
    }

    response = client.models.generate_content(model=params["model"],
                                              contents=[prompt],
                                              config=generation_config)

    # print(response.text)
    return response.text


def get_task_configuration(task: QATask):
    p = prompt.format(question=task.question)
    res = gemini_api_call_with_config(model_name="gemini-2.5-pro", prompt=p)
    print(res)

    configuration = parse_workflow_to_json(workflow_str=res)
    print(configuration)
    return configuration


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    tasks = load_test_dataset(
        path="/home/jiayuan/nl2sql/MultiHop-RAG/dataset/qa_sampled.json")

    # Test single task first
    start = time.time()
    get_task_configuration(tasks[0])
    print(f"time:{time.time()-start}")
    exit(0)
    # tasks = tasks[:3]

    configurations = [None] * len(tasks)  # Pre-allocate list with correct size

    def process_task(idx, task):
        """Process a single task and return index and result."""
        print(f"Processing task {idx}...")
        config = get_task_configuration(task)
        return idx, config

    # Use ThreadPoolExecutor with max_workers threads
    max_workers = 100  # Adjust based on your system

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_task, idx, task): idx
            for idx, task in enumerate(tasks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, config = future.result()
            configurations[idx] = {"id": idx, "configuration": config}
            print(f"Completed task {idx}")

    # Save to JSON file (already sorted by id)
    with open("configurations_2.json", "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(configurations)} configurations to configurations.json")
