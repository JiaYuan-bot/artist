prompt = """
You are a Text-to-SQL Workflow Planner.
Your job is to plan (not execute) an end-to-end Text-to-SQL workflow for a single query.
Select which modules to run (you may skip modules) and choose their configurations to minimize latency while meeting a quality threshold.

I will give user query, user hint, database schema and detailed workflow modules meta information with their condidate configurations. 
You should think carefully, output a workflow that you think it is the best fit for the user query. The workflow should minimize latency while meeting a quality threshold.

Inputs:
1. User query
2. User hint
3. DB schema


Workflow Modules(Each module has functional description and allowed configs. Also, we will give the skippable modules):

information_retrieval:
  description: Retrieve relevant database catalog information and value entities
  configs: 
    top_k: [3, 5, 10]
  
schema_linking: 
  description: Identify relavent tables/columns and filter out irrelevant tables/columns
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
  
candidate_generation:
  description: Generate SQL candidates
  configs:
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
    prompt: [basic, cot]

revison: 
  description: Execute SQL; if errors, repair & retry.
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]

evaluation: 
  description: Score the generated SQL
  configs:


All skippable cases:
[
  (information_retrieval),
  (schema_linking),
  (revision),
  (information_retrieval, schema_linking)
  (information_retrieval, revison),
  (schema_linking, revison)
  (information_retrieval, schema_linking, revison),
]

(information_retrieval) means skipping only information_retrieval
(information_retrieval, schema_linking, revison) means skip information_retrieval, schema_linking, revison


  
Rules and Heuristics

0. Carefully analysis the user query complexity(with hint, db schema), simpler indicate a light-weight workflow.
1. Prefer the cheapest workflow that plausibly generate correct SQL.
2. Skip aggressively when safe (small schema, simple query). But when you think this query is complex, don't be too aggressive. Keep safe!
3. Prefer small/medium LLM; escalate only if risk of failing the threshold is high.
4. Just-in-time activation: decide knobs for a module only if you include it.
5. Respect permit skippable modules (e.g., cannot skip candidate generation and evaluation).
7. Don't generate SQL here; plan the workflow only.
8. The calling order of modules cannot be disrupted

Output format, return a single object matching this example schema:

information_retrieval:
  configs: 
    top_k: 3
  
schema_linking: 
  configs: 
    LLM: gemini-2.5-pro
    temperature: 1.0
  
candidate_generation:
  configs:
    LLM: gemini-2.5-flash-lite
    temperature: 0.8
    prompt: basic

revison: 
  configs: 
    LLM: gemini-2.5-flash
    temperature: 0.5

evaluation: 
  configs:


Example

User query: List school names and mailing ZIPs under Avetik Atoian’s administration.
User hit: Avetik and Atoian are names
DB schema:
CREATE TABLE frpm
(
    CDSCode                                       TEXT not null
        primary key,
    `Academic Year`                               TEXT  null,
    `County Code`                                 TEXT  null,
    `District Code`                               INTEGER         null,
    `School Code`                                 TEXT  null,
    `County Name`                                 TEXT null,
    `District Name`                               TEXT null,
    `School Name`                                 TEXT null,
    `District Type`                               TEXT null,
    `School Type`                                 TEXT null,
    `Educational Option Type`                     TEXT null,
    `NSLP Provision Status`                       TEXT null,
    `Charter School (Y/N)`                        INTEGER    null,
    `Charter School Number`                       TEXT  null,
    `Charter Funding Type`                        TEXT null,
    IRC                                           INTEGER    null,
    `Low Grade`                                   TEXT  null,
    `High Grade`                                  TEXT null,
    `Enrollment (K-12)`                           REAL      null,
    `Free Meal Count (K-12)`                      REAL       null,
    `Percent (%) Eligible Free (K-12)`            REAL       null,
    `FRPM Count (K-12)`                           REAL       null,
    `Percent (%) Eligible FRPM (K-12)`            REAL       null,
    `Enrollment (Ages 5-17)`                      REAL       null,
    `Free Meal Count (Ages 5-17)`                 REAL       null,
    `Percent (%) Eligible Free (Ages 5-17)`       REAL       null,
    `FRPM Count (Ages 5-17)`                      REAL       null,
    `Percent (%) Eligible FRPM (Ages 5-17)`       REAL       null,
    `2013-14 CALPADS Fall 1 Certification Status` INTEGER    null,
    foreign key (CDSCode) references schools (CDSCode)
);
CREATE TABLE satscores
(
    cds         TEXT not null
        primary key,
    rtype       TEXT  not null,
    sname       TEXT null,
    dname       TEXT null,
    cname       TEXT null,
    enroll12    INTEGER         not null,
    NumTstTakr  INTEGER          not null,
    AvgScrRead  INTEGER          null,
    AvgScrMath  INTEGER          null,
    AvgScrWrite INTEGER          null,
    NumGE1500   INTEGER          null,
--     PctGE1500   double      null,
        foreign key (cds) references schools (CDSCode)
);
CREATE TABLE schools
(
    CDSCode     TEXT not null
        primary key,
    NCESDist    TEXT  null,
    NCESSchool  TEXT  null,
    StatusType  TEXT  not null,
    County      TEXT not null,
    District    TEXT not null,
    School      TEXT null,
    Street      TEXT null,
    StreetAbr   TEXT null,
    City        TEXT null,
    Zip         TEXT null,
    State       TEXT  null,
    MailStreet  TEXT null,
    MailStrAbr  TEXT null,
    MailCity    TEXT null,
    MailZip     TEXT null,
    MailState   TEXT  null,
    Phone       TEXT null,
    Ext         TEXT  null,
    Website     TEXT null,
    OpenDate    DATE        null,
    ClosedDate  DATE        null,
    Charter     INTEGER    null,
    CharterNum  TEXT  null,
    FundingType TEXT null,
    DOC         TEXT  not null,
    DOCType     TEXT not null,
    SOC         TEXT  null,
    SOCType     TEXT null,
    EdOpsCode   TEXT  null,
    EdOpsName   TEXT null,
    EILCode     TEXT  null,
    EILName     TEXT null,
    GSoffered   TEXT null,
    GSserved    TEXT  null,
    Virtual     TEXT  null,
    Magnet      INTEGER   null,
    Latitude    REAL      null,
    Longitude   REAL      null,
    AdmFName1   TEXT null,
    AdmLName1   TEXT null,
    AdmEmail1   TEXT null,
    AdmFName2   TEXT null,
    AdmLName2   TEXT null,
    AdmEmail2   TEXT null,
    AdmFName3   TEXT  null,
    AdmLName3   TEXT null,
    AdmEmail3   TEXT null,
    LastUpdate  DATE        not null
);



Planner output:

candidate_generation:
  configs:
    LLM: gemini-2.5-flash-lite
    temperature: 0.8
    prompt: basic

evaluation: 
  configs:

---
Now I will give the user query, user hint, db schema. Please give me the workflow.
User query {query}

User hint {hint}

DB schema {schema}

Planner output:
"""

import os
import json
import sqlite3

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

DB_DIR = "/home/jiayuan/nl2sql/chess_function_service/data/dev/dev_databases/"

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
class Task:
    """
    Represents a task with question and database details.

    Attributes:
        question_id (int): The unique identifier for the question.
        db_id (str): The database identifier.
        question (str): The question text.
        evidence (str): Supporting evidence for the question.
        SQL (Optional[str]): The SQL query associated with the task, if any.
        difficulty (Optional[str]): The difficulty level of the task, if specified.
    """
    question_id: int = field(init=False)
    db_id: str = field(init=False)
    question: str = field(init=False)
    evidence: str = field(init=False)
    SQL: Optional[str] = field(init=False, default=None)

    # difficulty: Optional[str] = field(init=False, default=None)

    def __init__(self, task_data: Dict[str, Any]):
        """
        Initializes a Task instance using data from a dictionary.

        Args:
            task_data (Dict[str, Any]): A dictionary containing task data.
        """
        self.question_id = task_data["question_id"]
        self.db_id = task_data["db_id"]
        self.question = task_data["question"]
        self.evidence = task_data["evidence"]
        self.SQL = task_data.get("SQL")
        # self.difficulty = task_data.get("difficulty")


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


def get_db_schema(db_path):
    """
    Get the schema of a SQLite database as SQL CREATE TABLE statements.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        str: Well-formatted SQL schema string
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    lines = []

    for table in tables:
        table_name = table[0]

        # Get the CREATE TABLE statement
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name, ))
        create_statement = cursor.fetchone()[0]

        lines.append(create_statement + ";")
        lines.append("")  # Empty line between tables

    conn.close()

    return "\n".join(lines)


def get_task_configuration(task: Task):
    db_path = Path(DB_DIR) / task.db_id / f"{task.db_id}.sqlite"
    schema = get_db_schema(db_path)
    p = prompt.format(query=task.question, hint=task.evidence, schema=schema)
    res = gemini_api_call_with_config(model_name="gemini-2.5-pro", prompt=p)
    print(res)

    configuration = parse_workflow_to_json(workflow_str=res)
    print(configuration)
    return configuration


def load_dataset(path):
    """
    Load a dataset from a JSON file containing a list of question objects.
    
    Args:
        path (str): Path to the JSON file
        
    Returns:
        list: List of dictionaries containing question data
    """
    with open(path, 'r', encoding='utf-8') as file:
        tasks = json.load(file)

    dataset = [Task(t) for t in tasks]
    # dataset = dataset[:1]  # Limit to first 1000 tasks for testing
    return dataset


def process_task(idx, task):
    """Process a single task and return its configuration with index"""
    config = get_task_configuration(task)
    return {"id": idx, "configuration": config}


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    import time
    configurations = []
    tasks = load_dataset(
        "/home/jiayuan/nl2sql/chess_function_service/data/dev/dev.json")
    # tasks = tasks[:3]
    start = time.time()
    process_task(0, tasks[0])
    print(f"time:{time.time()-start}")
    exit(0)
    # Use ThreadPoolExecutor to process tasks in parallel
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Submit all tasks with their indices
        futures = [
            executor.submit(process_task, idx, task)
            for idx, task in enumerate(tasks)
        ]

        # Collect results in order (maintains task0, task1, task2, ...)
        configurations = [future.result() for future in futures]

    # Save to JSON file
    with open("configurations.json", "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(configurations)} configurations to configurations.json")
