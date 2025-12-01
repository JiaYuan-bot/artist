import logging
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import os
import duckdb
import seaborn as sns
import sqlglot
import re
import time
import abc
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from pathlib import Path
from tqdm import trange
from func_timeout import FunctionTimedOut

from llm.models import gemini_api_call_with_config
from .utils import node_decorator, get_last_node_result, get_node_result
from .utils import parse_response, validate_select_order, add_group_by, parse_code_from_string,\
    parse_vql_from_string, extract_world_info, is_email, is_valid_date_column
from .const import *

description = "Translate VQL to python language using library{matplotlib, seaborn},and execute to perform validation"

DATA_PATH = "/home/jiayuan/nl2sql/nvAgent/visEval_dataset/databases/"


@node_decorator()
def translator(
    task: Any,
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
):
    """
    :param task:{
                    "db_id": database_name,
                    "query": user_query,
                    "tables": original tables information(table name)
                }
    :return: 
    """
    logging.info("Starting translator")

    data_path = DATA_PATH
    db_id, query = task.get('db_id'), task.get('query')
    vql = get_last_node_result(execution_history)['final_vql']
    db_info = ''

    if get_node_result(execution_history, 'processor'):
        db_info = get_node_result(execution_history,
                                  'processor').get('new_schema')
    else:
        db_info = get_node_result(execution_history,
                                  'preprocess').get('old_schema')

    db_path = f"{data_path}/{db_id}"
    library = task.get('library', 'matplotlib')
    code = translate(db_path, vql, library)
    # print(code)
    translator_result = {}
    translator_result['code'] = code

    logging.info("Finish translator")
    return translator_result

    # code = message.get('pred', self._translate(db_path, vql, library))

    # # without validator
    # message['pred'] = code
    # message['send_to'] = SYSTEM_NAME
    # return

    validator_result = {}
    # do not fix vql containing "error" string
    if 'error!' in vql:
        validator_result['code'] = code
        return validator_result

    is_timeout = False
    try:
        exec_result = execute_python_code(code)
        # logging.info(exec_result)
    except Exception as e:
        logging.error(e)
        exec_result = {'output': '', 'error': ''}
        is_timeout = True
    except FunctionTimedOut as fto:
        logging.error(fto)
        exec_result = {'output': '', 'error': ''}
        is_timeout = True

    is_need_refine = is_need_refine_func(exec_result)

    if not is_need_refine:
        if not validate_select_order(vql):
            is_need_refine = True
            exec_result[
                'error'] = "Incorrect select column numbers! If there are 3 columns, please use STACKED BAR, GROUPED LINE, or GROUPED SCATTER"

    if is_timeout:
        validator_result['code'] = code
    elif not is_need_refine:
        validator_result['code'] = code
    else:
        llm_config = config.get("llm_config", {})
        model_name = llm_config.get("model", "gemini-2.5-pro")
        new_vql = refine_vql(query, vql, db_info, exec_result, model_name)
        # validator_result['final_vql'] = new_vql
        # try again using use vql
        code = translate(db_path, new_vql, library)
        # validator_result['fixed'] = True
        validator_result['code'] = code

    logging.info("Finish validator")

    return validator_result


def translate_plus(db_path: str,
                   vql: str,
                   library="matplotlib"
                   ):  # translate vql to python code, with stacked bar chart
    try:

        vis_match = re.search(r'Visualize\s+([\w\s]+)\s+SELECT', vql,
                              re.IGNORECASE)
        if vis_match:
            vis_type = vis_match.group(1).upper().strip()
        else:
            raise ValueError("Visualization type not found in VQL")

        bin_clause = re.search(r'BIN\s+(.*?)\s+BY\s+(\w+)', vql, re.IGNORECASE)

        sql_match = re.search(r'SELECT\s+.+', vql, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0)
        else:
            raise ValueError("SQL query not found in VQL")

        sql = re.sub(r'\s+BIN\s+.*?BY\s+\w+', '', sql)
        parsed_sql = sqlglot.parse_one(sql)

        select = parsed_sql.find(sqlglot.exp.Select)
        if not select:
            raise ValueError("SELECT statement not found in SQL")

        select_exprs = select.expressions
        x_col = select_exprs[0].alias_or_name
        group_col = select_exprs[2].alias_or_name
        y_col = select_exprs[1].alias_or_name

        if isinstance(select_exprs[-1], sqlglot.exp.AggFunc):
            agg_func = select_exprs[-1].key.lower()
            agg_arg = select_exprs[-1].this.name if hasattr(
                select_exprs[-1].this, 'name') else str(select_exprs[-1].this)
            group_col = f"{agg_func}_{agg_arg}"
            select_exprs[-1] = select_exprs[-1].as_(group_col)
        if isinstance(select_exprs[1], sqlglot.exp.AggFunc):
            agg_func = select_exprs[1].key.lower()
            agg_arg = select_exprs[1].this.name if hasattr(
                select_exprs[1].this, 'name') else str(select_exprs[1].this)
            y_col = f"{agg_func}_{agg_arg}"
            select_exprs[1] = select_exprs[1].as_(y_col)

        sql = parsed_sql.sql()
        sql = add_group_by(sql, group_col)

        bin_code = ''
        if bin_clause:
            bin_col, bin_type = bin_clause.groups()
            x_col = bin_col

            sql = add_group_by(sql, bin_col)

            bin_code += "# Apply binning operation\n"
            bin_code += "flag = True\n"

            if bin_type.upper() == 'YEAR':
                bin_code += f"""
is_datetime = pd.api.types.is_datetime64_any_dtype(df['{bin_col}'])
if is_datetime:
    df['{x_col}'] = df['{x_col}'].dt.year
else:
    df['{x_col}'] = df['{x_col}'].astype(int)
    flag = False
"""
            elif bin_type.upper() == 'MONTH':
                bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                bin_code += f"df['{x_col}'] = df['{x_col}'].dt.strftime('%B')\n"
            elif bin_type.upper() == 'DAY':
                bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                bin_code += f"df['{x_col}'] = df['{x_col}'].dt.date()\n"
            elif bin_type.upper() == 'WEEKDAY':
                bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                bin_code += f"df['{x_col}'] = df['{x_col}'].dt.day_name()\n"

            parsed_sql = sqlglot.parse_one(sql)
            select_expr = parsed_sql.find(sqlglot.exp.Select)

            agg_func = 'size'

            for expr in select_expr.expressions:
                if isinstance(expr, sqlglot.exp.Alias) and expr.alias == y_col:
                    if isinstance(expr.this, sqlglot.exp.Count):
                        agg_func = 'size'
                    elif isinstance(expr.this, sqlglot.exp.Sum):
                        agg_func = 'sum'
                    elif isinstance(expr.this, sqlglot.exp.Avg):
                        agg_func = 'mean'

            if agg_func == 'size':
                bin_code += f"""
# Group by and calculate count
if flag:
    df = df.groupby(['{x_col}', '{group_col}']).sum().reset_index()
"""
            if agg_func == 'sum':
                bin_code += f"""
# Group by and calculate sum
if flag:
    df = df.groupby(['{x_col}', '{group_col}']).sum().reset_index()
"""
            if agg_func == 'mean':
                bin_code += f"""
# Group by and calculate avg
if flag:
    df = df.groupby(['{x_col}', '{group_col}']).mean().reset_index()
"""

            if bin_type.upper() == 'WEEKDAY':
                bin_code += f"""
# Ensure all seven days of the week are included
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

all_combinations = pd.MultiIndex.from_product([weekday_order, df['{group_col}'].unique()], 
                                              names=['{x_col}', '{group_col}'])

df = df.set_index(['{x_col}', '{group_col}'])
df = df.reindex(all_combinations, fill_value=0).reset_index()
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=weekday_order, ordered=True)
"""
            elif bin_type.upper() == 'MONTH':
                bin_code += f"""
# Sort months in chronological order, but only include existing months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
existing_months = df['{x_col}'].unique()
ordered_existing_months = [month for month in month_order if month in existing_months]
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=ordered_existing_months, ordered=True)
"""

            order_by = parsed_sql.find(sqlglot.exp.Order)
            if order_by:
                sort_columns = []
                sort_ascending = []
                for expr in order_by.expressions:

                    original_col_name = expr.this.name if isinstance(
                        expr.this, sqlglot.exp.Column) else str(expr.this)

                    if original_col_name == bin_col:
                        col_name = x_col
                    elif isinstance(expr.this, sqlglot.exp.AggFunc):

                        agg_func = expr.this.key.lower()
                        agg_arg = expr.this.this.name if hasattr(
                            expr.this.this, 'name') else str(expr.this.this)
                        col_name = f"{agg_func}_{agg_arg}"
                    else:
                        col_name = original_col_name
                    sort_columns.append(col_name)

                    sort_ascending.append(
                        expr.args.get('desc', False) == False)

                sort_columns_str = ", ".join(
                    [f"'{col}'" for col in sort_columns])
                sort_ascending_str = ", ".join(map(str, sort_ascending))

                bin_code += f"""
# Ensure sorting columns exist in the DataFrame
sort_columns = [{sort_columns_str}]
sort_columns = [col for col in sort_columns if col in df.columns]
if sort_columns:
    df = df.sort_values(sort_columns, ascending=[{sort_ascending_str}])
else:
    print("Warning: Specified sorting columns not found in the DataFrame. No sorting applied.")
    df = df.sort_values(['{group_col}', '{x_col}'])
"""
            else:

                bin_code += f"df = df.sort_values(['{group_col}', '{x_col}'])\n"

        pivot = False
        vis_code = ""
        if "BAR" in vis_type:
            if library == 'matplotlib':
                pivot = True

                vis_code += f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
df_pivot = df.pivot(index='{x_col}', columns='{group_col}', values='{y_col}')
df_pivot.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Stacked Bar Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
            elif library == 'seaborn':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.barplot(x='{x_col}', y='{y_col}', hue='{group_col}', data=df, ax=ax, alpha=0.8)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Stacked Bar Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
        elif "LINE" in vis_type:
            if library == 'matplotlib':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for group in df['{group_col}'].unique():
    group_data = df[df['{group_col}'] == group]
    ax.plot(group_data['{x_col}'], group_data['{y_col}'], label=group, marker='o', alpha=0.7)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Line Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
            elif library == 'seaborn':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.lineplot(x='{x_col}', y='{y_col}', hue='{group_col}', data=df, marker='o', ax=ax, alpha=0.7)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Line Chart of {y_col} by {x_col} and {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
        elif "SCATTER" in vis_type:
            if library == 'matplotlib':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for group in df['{group_col}'].unique():
    group_data = df[df['{group_col}'] == group]
    ax.scatter(group_data['{x_col}'], group_data['{y_col}'], label=group, alpha=0.6, s=80, edgecolor='k')
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Scatter Plot of {y_col} vs {x_col} by {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""
            elif library == 'seaborn':
                vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.scatterplot(x='{x_col}', y='{y_col}', hue='{group_col}', data=df, ax=ax, alpha=0.6, s=80, edgecolor='k')
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('Grouped Scatter Plot of {y_col} vs {x_col} by {group_col}')
ax.legend(title='{group_col}', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
"""

        python_code = ''
        if library == "seaborn":
            python_code += "import seaborn as sns\n"
        python_code += f"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb


data_folder = '{db_path}'


con = duckdb.connect(database=':memory:')


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {{table_name}} AS SELECT * FROM read_csv_auto('{{os.path.join(data_folder, file)}}')")


sql = f'''
{sql}
'''
df = con.execute(sql).fetchdf()


df.columns = ['{x_col}', '{y_col}', '{group_col}']


# print("Columns in the dataframe:", df.columns)


{bin_code}


{vis_code}
"""
        if pivot:
            python_code += f"""
# Print data
x_data = [df.index.tolist()]
y_data = [df[col].tolist() for col in df.columns]
print("x_data:", x_data)
print("y_data:", y_data)
print("groups:", df.columns.tolist())
"""
        else:
            python_code += f"""
# Print data
print("x_data (unique):", df['{x_col}'].unique().tolist())
print("group_data (unique):", df['{group_col}'].unique().tolist())
print("y_data (sum for each group):", df.groupby(['{x_col}', '{group_col}'])['{y_col}'].sum().to_dict())
"""
        return python_code

    except Exception as e:
        print(f"Error in _translate function: {e}")
        print(f"VQL: {vql}")
        print(f"Parsed SQL: {sql}")

        return f"Error occurred while processing the query: {str(e)}"


def translate_normal(
    db_path: str,
    vql: str,
    library="matplotlib"
):  # translate vql to python code using matplotlib or seaborn
    try:

        vis_type = re.search(r'Visualize\s+(\w+)', vql, re.IGNORECASE).group(1)

        bin_clause = re.search(r'BIN\s+(.*?)\s+BY\s+(\w+)', vql, re.IGNORECASE)

        sql = re.sub(r'Visualize\s+\w+\s+', '', vql)
        sql = re.sub(r'\s+BIN\s+.*?BY\s+\w+', '', sql)
        parsed_sql = sqlglot.parse_one(sql)

        select = parsed_sql.find(sqlglot.exp.Select)
        if not select:
            raise ValueError("SELECT statement not found in SQL")

        select_exprs = select.expressions
        if len(select_exprs) < 2:
            raise ValueError(
                f"Not enough expressions in SELECT statement. Found {len(select_exprs)}, expected at least 2"
            )
        # print(select_exprs)
        x_col = select_exprs[0].alias_or_name
        y_col = select_exprs[1].alias_or_name

        if isinstance(select_exprs[1], sqlglot.exp.AggFunc):
            agg_func = select_exprs[1].key.lower()
            agg_arg = select_exprs[1].this.name if hasattr(
                select_exprs[1].this, 'name') else str(select_exprs[1].this)
            y_col = f"{agg_func}_{agg_arg}"
            select_exprs[1] = select_exprs[1].as_(y_col)

        sql = parsed_sql.sql()

        bin_code = ''
        if bin_clause:
            bin_col, bin_type = bin_clause.groups()
            x_col = bin_col

            sql = add_group_by(sql, bin_col)

            bin_code += "# Apply binning operation\n"
            bin_code += "flag = True\n"

            if bin_type.upper() == 'YEAR':
                bin_code += f"""
is_datetime = pd.api.types.is_datetime64_any_dtype(df['{bin_col}'])
if is_datetime:
    df['{x_col}'] = df['{x_col}'].dt.year
else:
    df['{x_col}'] = df['{x_col}'].astype(int)
    flag = False
"""
            elif bin_type.upper() == 'MONTH':
                bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                bin_code += f"df['{x_col}'] = df['{x_col}'].dt.strftime('%B')\n"
            elif bin_type.upper() == 'DAY':
                bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                bin_code += f"df['{x_col}'] = df['{x_col}'].dt.date\n"
            elif bin_type.upper() == 'WEEKDAY':
                bin_code += f"df['{bin_col}'] = pd.to_datetime(df['{bin_col}'])\n"
                bin_code += f"df['{x_col}'] = df['{x_col}'].dt.day_name()\n"

            parsed_sql = sqlglot.parse_one(sql)
            select_expr = parsed_sql.find(sqlglot.exp.Select)

            agg_func = 'size'

            for expr in select_expr.expressions:
                if isinstance(expr, sqlglot.exp.Alias) and expr.alias == y_col:
                    if isinstance(expr.this, sqlglot.exp.Count):
                        agg_func = 'size'
                    elif isinstance(expr.this, sqlglot.exp.Sum):
                        agg_func = 'sum'
                    elif isinstance(expr.this, sqlglot.exp.Avg):
                        agg_func = 'mean'

            if agg_func == 'size':
                bin_code += f"""
# Group by and calculate count
if flag:
    df = df.groupby('{x_col}').sum().reset_index()
"""
            if agg_func == 'sum':
                bin_code += f"""
# Group by and calculate sum
if flag:
    df = df.groupby('{x_col}').sum().reset_index()
"""
            if agg_func == 'mean':
                bin_code += f"""
# Group by and calculate avg
if flag:
    df = df.groupby('{x_col}').mean().reset_index()
"""

            if bin_type.upper() == 'WEEKDAY':
                bin_code += f"""
# Ensure all seven days of the week are included
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df = df.set_index('{x_col}').reindex(weekday_order, fill_value=0).reset_index()
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=weekday_order, ordered=True)
"""

            order_by = parsed_sql.find(sqlglot.exp.Order)
            if order_by:
                sort_columns = []
                sort_ascending = []
                for expr in order_by.expressions:

                    original_col_name = expr.this.name if isinstance(
                        expr.this, sqlglot.exp.Column) else str(expr.this)

                    if original_col_name == bin_col:
                        col_name = x_col
                    elif isinstance(expr.this, sqlglot.exp.AggFunc):

                        agg_func = expr.this.key.lower()
                        agg_arg = expr.this.this.name if hasattr(
                            expr.this.this, 'name') else str(expr.this.this)
                        col_name = f"{agg_func}_{agg_arg}"
                    else:
                        col_name = original_col_name
                    sort_columns.append(col_name)

                    sort_ascending.append(
                        expr.args.get('desc', False) == False)

                sort_columns_str = ", ".join(
                    [f"'{col}'" for col in sort_columns])
                sort_ascending_str = ", ".join(map(str, sort_ascending))

                bin_code += f"""
# Ensure sorting columns exist in the DataFrame
sort_columns = [{sort_columns_str}]
sort_columns = [col for col in sort_columns if col in df.columns]
if sort_columns:
    df = df.sort_values(sort_columns, ascending=[{sort_ascending_str}])
else:
    print("Warning: Specified sorting columns not found in the DataFrame. No sorting applied.")
    df = df.sort_values('{x_col}')
"""
            else:

                if bin_type.upper() == 'MONTH':
                    bin_code += f"""
# Sort months in chronological order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
df['{x_col}'] = pd.Categorical(df['{x_col}'], categories=month_order, ordered=True)
df = df.sort_values('{x_col}')
"""
                elif bin_type.upper() == 'WEEKDAY':
                    bin_code += f"df = df.sort_values('{x_col}')\n"
                else:
                    bin_code += f"df = df.sort_values('{x_col}')\n"

        vis_code = ""
        if library == 'matplotlib':
            vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
{"ax.bar(df['" + x_col + "'], df['" + y_col + "'])" if vis_type == 'BAR' else ""}
{"ax.plot(df['" + x_col + "'], df['" + y_col + "'])" if vis_type == 'LINE' else ""}
{"ax.scatter(df['" + x_col + "'], df['" + y_col + "'])" if vis_type == 'SCATTER' else ""}
{"ax.pie(df['" + y_col + "'], labels=df['" + x_col + "'], autopct='%1.1f%%')" if vis_type == 'PIE' else ""}
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title(f'{vis_type} Chart of {y_col} by {x_col}')
{"plt.xticks(rotation=45)" if vis_type != 'PIE' else ""}
plt.tight_layout()
"""

        elif library == 'seaborn':
            vis_code = f"""
fig,ax = plt.subplots(1,1,figsize=(10,4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
{"sns.barplot(x='" + x_col + "', y='" + y_col + "', data=df, ax=ax)" if vis_type == 'BAR' else ""}
{"sns.lineplot(x='" + x_col + "', y='" + y_col + "', data=df, ax=ax)" if vis_type == 'LINE' else ""}
{"sns.scatterplot(x='" + x_col + "', y='" + y_col + "', data=df, ax=ax)" if vis_type == 'SCATTER' else ""}
{"ax.pie(df['" + y_col + "'], labels=df['" + x_col + "'], autopct='%1.1f%%')" if vis_type == 'PIE' else ""}
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('{vis_type} Chart of {y_col} by {x_col}')
{"plt.xticks(rotation=45)" if vis_type != 'PIE' else ""}
sns.despine()
plt.tight_layout()
"""

        python_code = ''
        if library == "seaborn":
            python_code += "import seaborn as sns\n"
        python_code += f"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb


data_folder = '{db_path}'


con = duckdb.connect(database=':memory:')


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {{table_name}} AS SELECT * FROM read_csv_auto('{{os.path.join(data_folder, file)}}')")


sql = f'''
{sql}
'''
df = con.execute(sql).fetchdf()
con.close()

# rename columns
df.columns = ['{x_col}','{y_col}']


# print("Columns in the dataframe:", df.columns)


{bin_code}


{vis_code}

# Print data
print("x_data:", df['{x_col}'].tolist())
print("y_data:", df['{y_col}'].tolist())
"""
        return python_code

    except Exception as e:
        print(f"Error in _translate function: {e}")
        print(f"VQL: {vql}")
        print(f"Parsed SQL: {sql}")

        return f"Error occurred while processing the query: {str(e)}"


def translate(db_path: str, vql: str, library="matplotlib"):
    try:
        match = re.search(r'Visualize\s+([\w\s]+)\s+SELECT\s+(.*?)\s+FROM',
                          vql, re.IGNORECASE | re.DOTALL)
        if not match:
            return False

        vis_type = match.group(1).upper().strip()
        select_columns = [col.strip() for col in match.group(2).split(',')]

        if vis_type in ["STACKED BAR", "GROUPED LINE", "GROUPED SCATTER"
                        ] or len(select_columns) == 3:
            return translate_plus(db_path, vql, library)
        else:
            return translate_normal(db_path, vql, library)
    except Exception as e:
        print("error in translate:", e)


def execute_python_code(code):

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {'output': '', 'error': ''}

    exec_globals = {}

    original_show = plt.show

    def dummy_show(*args, **kwargs):
        pass

    plt.show = dummy_show

    try:

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):

            exec(code, exec_globals)

        result['output'] = stdout_capture.getvalue()

    except Exception as e:

        result[
            'error'] = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

    finally:

        plt.show = original_show

    result['error'] += stderr_capture.getvalue()

    return result


def is_need_refine_func(exec_result: dict):
    flag = False
    if exec_result['error']:
        flag = True
        if "UserWarning: set_ticklabels()" in exec_result['error']:
            flag = False
        if "RuntimeWarning" in exec_result['error']:
            flag = False
        if "UserWarning: Tight layout not applied" in exec_result['error']:
            flag = False
        if "FutureWarning" in exec_result['error']:
            flag = False
    return flag


def refine_vql(nl_query: str, vql: str, db_info, exec_result: dict,
               model: str):
    error = exec_result['error']
    prompt = refiner_vql_template.format(query=nl_query,
                                         db_info=db_info,
                                         vql=vql,
                                         error=error)
    reply = gemini_api_call_with_config(model, prompt)
    new_vql = parse_vql_from_string(reply)
    return new_vql


def refine_python(nl_query: str, code: str, db_info, exec_result: dict,
                  model: str):
    error = exec_result['error']
    prompt = refiner_python_template.format(query=nl_query,
                                            db_info=db_info,
                                            code=code,
                                            error=error)

    reply = gemini_api_call_with_config(model, prompt)
    new_code = parse_code_from_string(reply)
    return new_code
