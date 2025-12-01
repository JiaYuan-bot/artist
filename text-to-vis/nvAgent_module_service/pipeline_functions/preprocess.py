import logging
from typing import Dict, List, Any

import pandas as pd
from pathlib import Path

from .utils import node_decorator, get_last_node_result
from .utils import is_email, is_valid_date_column
from .const import *


@node_decorator()
def preprocess(
    task: Any,
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
):
    """
    :param task: {"db_id": database_name,
                    "query": user_query,
                    "tables": table names
    :return: database schema description str
    """
    logging.info("Starting preprocess")

    tables = task.get('tables')
    db_schema = get_db_desc_str(tables)

    result = {'old_schema': db_schema}

    logging.info("Finish preprocess")

    return result


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
