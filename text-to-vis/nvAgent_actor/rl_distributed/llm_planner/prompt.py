prompt = """
You are a Text-to-Visualization Workflow Planner.
Your job is to plan (not execute) an end-to-end Text-to-Visualization workflow for a single query.
Select which modules to run (you may skip modules) and choose their configurations to minimize latency while meeting a quality threshold.

I will give user query, database schema and detailed workflow modules meta information with their condidate configurations. 
You should think carefully, output a workflow that you think it is the best fit for the user query. The workflow should minimize latency while meeting a quality threshold.

Inputs:
1. User query
3. DB schema


Workflow Modules(Each module has functional description and allowed configs. Also, we will give the skippable modules):

preprocess:
  description: load database schema
  configs:

processor:
  description: filter out irrelevant tables/columns; generate database schema augmented explanation; determine input difficulty
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
  
composer: 
  description: generate a valid VQL (Visualization Query Language) sentence
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
    prompt: [basic, cot]

translator:
  description: translate VQL to python code which will generate visualization
  configs: 
    
validator: 
  description: Execute python code; if errors, repair & retry.
  configs: 
    LLM: [gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite]
    temperature: [0.5, 0.8, 1.0]
    prompt: [basic, cot]

evaluation: 
  description: Score the generated visualization
  configs:


All skippable cases:
[
  (processor),
  (validator),
  (processor, validator),
]

(processor) means skipping only processor
(processor, validator) means skip processor, validator


  
Rules and Heuristics

0. Carefully analysis the user query complexity(with db schema), simpler indicate a light-weight workflow.
1. Prefer the cheapest workflow that plausibly generate correct VQL.
2. Skip aggressively when safe (small schema, simple query). But when you think this query is complex, don't be too aggressive. Keep safe!
3. Prefer small/medium LLM; escalate only if risk of failing the threshold is high.
4. Just-in-time activation: decide knobs for a module only if you include it.
5. Respect permit skippable modules (e.g., cannot composer, translator).
7. Don't generate VQL here; plan the workflow only.
8. The calling order of modules cannot be disrupted

Output format! return a single object matching this example schema, do not output json format!

preprocess:
  configs:

processor:
  configs: 
    LLM: gemini-2.5-pro
    temperature: 0.8
  
composer: 
  configs: 
    LLM: gemini-2.5-pro
    temperature: 1.0
    prompt: basic

translator:
  configs: 
    
validator: 
  configs: 
    LLM: gemini-2.5-flash
    temperature: 1.0
    prompt: basic

nv_evaluation: 
  configs:


Example

User query: Show me a pie chart for what are the nationalities and the total ages of journalists?
DB schema:
# Table: journalist, (journalist)
[
  (journalist_ID, journalist id, Value examples: [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]. And this is an id type column),
  (Name, name, Value examples: ['Frank Mitcheson', 'Fred Chandler', 'Fred Keenor', 'George Gilchrist', 'Herbert Swindells', 'Jack Meaney'].),
  (Nationality, nationality, Value examples: ['England', 'Northern Ireland', 'Wales'].),
  (Age, age, Value examples: [np.int64(28), np.int64(43), np.int64(37), np.int64(25), np.int64(29), np.int64(27)].),
  (Years_working, years working, Value examples: [np.int64(6), np.int64(1), np.int64(3), np.int64(5), np.int64(7), np.int64(8)].)
]


Planner output:

preprocess:
  configs:

composer: 
  configs: 
    LLM: gemini-2.5-pro
    temperature: 1.0
    prompt: basic

translator:
  configs: 

nv_evaluation: 
  configs:

---
Now I will give the user query, user hint, db schema. Please give me the workflow.
User query {query}

DB schema {schema}

Planner output:
"""

import os
import json
import sqlite3

import re
import json
from typing import Dict
import sqlglot
import logging
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
class NVTask:
    """
    Represents a task with question and database details.

    Attributes:
        id (int): The unique identifier for the question.
        db_id (str): The database identifier.
        query (str): The question text.
        tables (list(str)): table names
        chart (str): vis type
        vis_obj (dict[str,ANY]): vis details
        query_meta (dict[str,ANY]): query meta information
    """
    id: int = field(init=False)
    db_id: str = field(init=False)
    query: str = field(init=False)
    tables: list[str] = field(init=False)
    chart: str = field(init=False)
    vis_obj: dict[str, Any] = field(init=False)
    query_meta: dict[str, Any] = field(init=False)

    # difficulty: Optional[str] = field(init=False, default=None)

    def __init__(self, task_data: Dict[str, Any]):
        """
        Initializes a Task instance using data from a dictionary.

        Args:
            task_data (Dict[str, Any]): A dictionary containing task data.
        """
        self.id = task_data["id"]
        self.db_id = task_data["db_id"]
        self.query = task_data["query"]
        self.tables = task_data["tables"]
        self.chart = task_data.get("chart")
        self.vis_obj = task_data.get("vis_obj")
        self.query_meta = task_data.get("query_meta")


#############   #############  #############
def load_table_info(table_path: str):
    table = pd.read_csv(table_path)
    table_name = Path(table_path).stem
    column_names = table.columns.tolist()
    column_types = [str(dtype) for dtype in table.dtypes]
    value_count = len(table)

    return {
        'table_name': table_name,
        'column_names': column_names,
        'column_types': column_types,
        'value_count': value_count
    }


def get_column_attributes(table_path):
    table = pd.read_csv(table_path)
    column_names = table.columns.tolist()
    column_types = [str(dtype) for dtype in table.dtypes]
    return column_names, column_types


def get_unique_column_values_str(table_path, column_names, column_types):
    table = pd.read_csv(table_path)
    col_to_values_str_lst = []
    col_to_values_str_dict = {}
    for idx, column_name in enumerate(column_names):

        lower_column_name: str = column_name.lower()
        # if lower_column_name ends with [id, email, url], just use empty str
        if lower_column_name.endswith('email') or \
                lower_column_name.endswith('url'):
            values_str = ''
            col_to_values_str_dict[column_name] = values_str
            continue

        grouped = table.groupby(column_name)
        group_counts = grouped.size()
        sorted_counts = group_counts.sort_values(ascending=False)
        values = sorted_counts.index.values
        dtype = sorted_counts.index.dtype

        values_str = ''
        # try to get value examples str, if exception, just use empty str
        try:
            values_str = get_value_examples_str(values, column_types[idx])
        except Exception as e:
            print(f"\nerror: get_value_examples_str failed, Exception:\n{e}\n")

        col_to_values_str_dict[column_name] = values_str

    for column_name in column_names:
        values_str = col_to_values_str_dict.get(column_name, '')
        col_to_values_str_lst.append([column_name, values_str])
    return col_to_values_str_lst


def get_value_examples_str(values: List[object], col_type: str):
    if not len(values):
        return ''

    vals = []
    has_null = False
    for v in values:
        if v is None:
            has_null = True
        else:
            tmp_v = str(v).strip()
            if tmp_v == '':
                continue
            else:
                vals.append(v)
    if not vals:
        return ''

    if len(values) > 10 and col_type in ['int64', 'float64']:
        vals = vals[:4]
        if has_null:
            vals.insert(0, None)
        return str(vals)

    # drop meaningless values of text type
    if col_type == 'object':
        new_values = []
        for v in vals:
            if not isinstance(v, str):
                new_values.append(v)
            else:
                if v == '':  # exclude empty string
                    continue
                elif ('https://' in v) or ('http://' in v):  # exclude url
                    return ''
                elif is_email(v):  # exclude email
                    return ''
                else:
                    new_values.append(v)
        vals = new_values
        tmp_vals = [len(str(a)) for a in vals]
        if not tmp_vals:
            return ''
        max_len = max(tmp_vals)
        if max_len > 50:
            return ''
    if not vals:
        return ''
    vals = vals[:6]
    is_date_column = is_valid_date_column(vals)
    if is_date_column:
        vals = vals[:1]
    if has_null:
        vals.insert(0, None)
    val_str = str(vals)
    return val_str


def load_db_info(tables: List[str]) -> dict:
    table2coldescription = {}
    table_unique_column_values = {}

    for table_path in tables:
        table_info = load_table_info(table_path)
        table_name = table_info['table_name']

        col2dec_lst = []
        all_column_names, all_column_types = get_column_attributes(table_path)
        col_values_str_lst = get_unique_column_values_str(
            table_path, all_column_names, all_column_types)
        table_unique_column_values[table_name] = col_values_str_lst

        for x, column_name in enumerate(all_column_names):
            lower_column_name = column_name.lower()
            column_desc = ''
            col_type = all_column_types[x]
            if lower_column_name.endswith('id'):
                column_desc = 'this is an id type column'
            elif lower_column_name.endswith('url'):
                column_desc = 'this is a url type column'
            elif lower_column_name.endswith('email'):
                column_desc = 'this is an email type column'
            elif table_info['value_count'] > 10 and col_type in [
                    'int64', 'float64'
            ] and col_values_str_lst[x][1] == '':
                column_desc = 'this is a number type column'

            full_col_name = column_name.replace('_', ' ').lower()
            col2dec_lst.append([full_col_name, column_desc])

        table2coldescription[table_name] = col2dec_lst

    result = {
        "desc_dict": table2coldescription,
        "value_dict": table_unique_column_values
    }
    return result


def build_table_schema_list_str(table_name, new_columns_desc, new_columns_val):

    table_desc: str = table_name.lower()
    table_desc = table_desc.replace('_', ' ')
    schema_desc_str = ''
    schema_desc_str += f"# Table: {table_name}, ({table_desc})\n"
    extracted_column_infos = []
    for (col_full_name, col_extra_desc), (col_name, col_values_str) in zip(
            new_columns_desc, new_columns_val):
        col_extra_desc = 'And ' + str(
            col_extra_desc) if col_extra_desc != '' and str(
                col_extra_desc) != 'nan' else ''
        col_extra_desc = col_extra_desc[:100]

        col_line_text = ''
        col_line_text += f'  ('
        col_line_text += f"{col_name}, "
        col_line_text += f"{col_full_name},"
        if col_values_str != '':
            col_line_text += f" Value examples: {col_values_str}."
        if col_extra_desc != '':
            col_line_text += f" {col_extra_desc}"
        col_line_text += '),'
        extracted_column_infos.append(col_line_text)
    schema_desc_str += '[\n' + '\n'.join(extracted_column_infos).strip(
        ',') + '\n]' + '\n'
    return schema_desc_str


def get_db_desc_str(tables: List[str]):
    db_info = load_db_info(tables)
    desc_info = db_info['desc_dict']
    value_info = db_info['value_dict']

    schema_desc_str = ''
    for table_name in desc_info.keys():
        columns_desc = desc_info[table_name]
        columns_val = value_info[table_name]
        new_columns_desc = columns_desc.copy()
        new_columns_val = columns_val.copy()

        schema_desc_str += build_table_schema_list_str(table_name,
                                                       new_columns_desc,
                                                       new_columns_val)

    return schema_desc_str.strip()


def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False


def is_valid_date_column(col_value_lst):
    for col_value in col_value_lst:
        if not is_valid_date(col_value):
            return False
    return True


def is_valid_date(date_str):
    if (not isinstance(date_str, str)):
        return False
    date_str = date_str.split()[0]
    if len(date_str) != 10:
        return False
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(pattern, date_str):
        year, month, day = map(int, date_str.split('-'))
        if year < 1 or month < 1 or month > 12 or day < 1 or day > 31:
            return False
        else:
            return True
    else:
        return False


#####
class Dataset:

    def __init__(
        self,
        folder: Path,
        table_type: str = "all",
        with_irrelevant_tables: bool = False,
    ):
        self.folder = folder
        dict_name = "visEval"
        if table_type in ["test", "train", "eva"]:
            dict_name += "_" + table_type
        dict_name += ".json"
        with open(folder / dict_name) as f:
            self.dict = json.load(f)

        with open(folder / "databases/db_tables.json") as f:
            self.db_tables = json.load(f)

        def benchmark():
            for key in list(self.dict.keys()):
                self.dict[key]["id"] = key
                self.dict[key]["tables"] = self.__get_tables(
                    key, with_irrelevant_tables)
                yield self.dict[key]

        self.benchmark = benchmark()

    def load_dataset(self) -> List[NVTask]:
        id = 0
        tasks = []
        for instance in self.benchmark:
            nl_queries = instance["nl_queries"]
            db_id = instance['db_id']
            tables = instance['tables']

            for index in range(len(nl_queries)):
                task_data = {
                    'id': id,
                    'db_id': db_id,
                    'query': nl_queries[index],
                    'tables': tables,
                    'chart': instance['chart'],
                    'vis_obj': instance['vis_obj'],
                    "query_meta": instance["query_meta"][index]
                }
                task = NVTask(task_data)
                tasks.append(task)
                id += 1
        return tasks

    def __get_tables(self, id: str, with_irrelevant_tables: bool = False):
        spec = self.dict[id]
        db_id = spec["db_id"]
        # table name
        all_table_names = self.db_tables[db_id]
        table_names = [
            x for x in all_table_names
            if x.lower() in spec["vis_query"]["VQL"].lower().split()
        ]

        if with_irrelevant_tables:
            irrelevant_tables = spec["irrelevant_tables"]
            table_names.extend(irrelevant_tables)

        tables = list(
            map(
                lambda table_name:
                f"{self.folder}/databases/{db_id}/{table_name}.csv",
                table_names,
            ))

        return tables


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


def get_task_configuration(task: NVTask):
    schema = get_db_desc_str(task.tables)
    p = prompt.format(query=task.query, schema=schema)
    res = gemini_api_call_with_config(model_name="gemini-2.5-pro", prompt=p)
    print(res)

    configuration = parse_workflow_to_json(workflow_str=res)
    print(configuration)
    return configuration


if __name__ == "__main__":

    configurations = []
    ds = Dataset(folder=Path("/home/jiayuan/nl2sql/nvAgent/visEval_dataset"),
                 table_type="eva")
    tasks = ds.load_dataset()
    import time
    start = time.time()
    get_task_configuration(tasks[0])
    print(f"time:{time.time()-start}")
    exit(0)

    # tasks = tasks[:3]
    for idx, task in enumerate(tasks):
        print(f"progress {idx}.........")
        config = get_task_configuration(task)  # Fixed: should be () not []
        configurations.append({"id": idx, "configuration": config})

    # Save to JSON file
    with open("configurations.json", "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(configurations)} configurations to configurations.json")
