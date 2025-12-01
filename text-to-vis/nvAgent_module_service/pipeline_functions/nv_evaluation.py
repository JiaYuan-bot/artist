import logging
from typing import Dict, List, Any

import matplotlib.pyplot as plt
from tqdm import trange
from typing import Optional
import base64
import logging
import os
from typing import Union
import traceback
import cairosvg
from attr import dataclass

from llm.engine_configs import ENGINE_CONFIGS
from .utils import node_decorator, get_last_node_result, get_node_result, show_svg
from .const import *

vision_model_config = ENGINE_CONFIGS['gemini-2.5-flash']
constructor = vision_model_config["constructor"]
params = vision_model_config["params"]
vision_model = constructor(**params)

from viseval.check import (
    chart_check,
    data_check,
    deconstruct,
    layout_check,
    order_check,
    readability_check,
    scale_and_ticks_check,
)


@dataclass
class CheckResult:
    answer: Union[bool, int]
    aspect: str
    rationale: str

    def get_json(self):
        return {
            "answer": self.answer,
            "aspect": self.aspect,
            "rationale": self.rationale,
        }


@dataclass
class ChartExecutionResult:
    """Response from a visualization execution"""

    # True if successful, False otherwise
    status: bool
    # Generate svg string if status is True
    svg_string: Optional[str] = None
    # Error message if status is False
    error_msg: Optional[str] = None


@node_decorator()
def nv_evaluation(
    task: Any,
    execution_history: Dict[str, Any],
    config: Dict[str, Any],
):
    """
    :param task:{
                    "db_id": database_name,
                    "query": user_query,
                    "tables": original tables information(table name)
                    ...
                    "chart":
                    "vis_obj":
                    "query_meta":
                }
    :return: 
    """
    logging.info("Starting nv_evaluation")

    nl_query = task.get("query")
    tables = task.get("tables")

    context = {}
    context["tables"] = tables
    context["library"] = task.get("library", "matplotlib")
    code = get_last_node_result(execution_history).get(
        'code', "import matplotlib.pyplot as plt")
    code += "\nplt.show()"
    # print(code)

    results = validity_check(code, context)

    pass_validity = all([result.answer for result in results])
    # print("pass_validity", pass_validity)
    if pass_validity:
        ground_truth = {
            "chart": task["chart"],
            "vis_obj": task["vis_obj"],
            "meta_info": task["query_meta"],
        }
        results += legality_check(context, ground_truth)

    pass_legality = all([result.answer for result in results])
    # print("pass_legality", pass_legality)
    if pass_legality:
        readability_results = readability_evaluate(context, nl_query)
        # print("readability", readability_results)
        results += readability_results

    evaluation_result = {}
    evaluation_result = {
        "pass":
        pass_legality,
        "pass_validity":
        pass_validity,
        "pass_legality":
        pass_legality,
        "pass_scale-and-ticks-check":
        readability_results[-2].answer if pass_legality else False,
        "readability_score":
        readability_results[-1].answer if pass_legality else 0,
    }
    print(evaluation_result)
    logging.info("Finish nv_evaluation")
    return evaluation_result


def surface_form_check(code) -> CheckResult:
    if "plt.show()" not in code:
        return CheckResult(
            answer=False,
            aspect="surface-form check",
            rationale="Did not plot visualization.",
        )
    else:
        return CheckResult(
            answer=True,
            aspect="surface-form check",
            rationale="Plotted visualization.",
        )


def validity_check(code, context) -> list[CheckResult]:
    results = []
    result = execute(code, context)
    results.append(result)
    if result.answer:
        result = surface_form_check(code)
        results.append(result)

    return results


def execute(code, context) -> CheckResult:
    result = execute_to_svg(code)
    if result.status is False:
        return CheckResult(answer=False,
                           aspect="code execution",
                           rationale=result.error_msg)

    context["svg_string"] = result.svg_string
    return CheckResult(
        answer=True,
        aspect="code execution",
        rationale="Code executed successfully.",
    )


def deconstruction(context) -> CheckResult:
    svg_string = context["svg_string"]
    library = context["library"]
    if library == "seaborn":
        library = "matplotlib"
    try:
        chart_info, msg = deconstruct(svg_string, library)
        if chart_info is None:
            return CheckResult(
                answer=False,
                aspect="deconstruction",
                rationale=msg,
            )
        context.update(chart_info)
        return CheckResult(
            answer=True,
            aspect="deconstruction",
            rationale="Deconstructed the chart successfully.",
        )
    except:
        return CheckResult(
            answer=False,
            aspect="deconstruction",
            rationale="Cannot parse the visualization.",
        )


def chart_type_check(context, ground_truth) -> CheckResult:
    answer, rationale = chart_check(
        context,
        ground_truth["chart"],
        (ground_truth["meta_info"]["stacked_bar"]
         if "stacked_bar" in ground_truth["meta_info"] else None),
    )
    return CheckResult(
        answer=answer,
        aspect="chart type check",
        rationale=rationale,
    )


def nv_data_check(context, ground_truth) -> CheckResult:
    answer, rationale = data_check(
        context,
        ground_truth["vis_obj"],
        ground_truth["meta_info"]["channel_specified"],
    )
    return CheckResult(
        answer=answer,
        aspect="data check",
        rationale=rationale,
    )


def nv_order_check(context, ground_truth) -> CheckResult:
    answer, rationale = order_check(
        context,
        ground_truth["vis_obj"],
        (ground_truth["meta_info"]["sort_by"]
         if "sort_by" in ground_truth["meta_info"] else None),
    )
    return CheckResult(
        answer=answer,
        aspect="order check",
        rationale=rationale,
    )


def legality_check(context, ground_truth) -> list[CheckResult]:
    results = []
    result = deconstruction(context)
    results.append(result)

    if result.answer:
        chart_type_check_result = chart_type_check(context, ground_truth)
        # print(chart_type_check_result.answer)
        data_check_result = nv_data_check(context, ground_truth)
        # print(data_check_result.answer)
        results.append(chart_type_check_result)
        results.append(data_check_result)
        if data_check_result.answer and ground_truth["vis_obj"][
                "sort"] is not None:
            nv_order_check(context, ground_truth)
            # print(nv_order_check(context, ground_truth).answer)
            results.append(nv_order_check(context, ground_truth))
    return results


def layout_check(context) -> CheckResult:
    assert "svg_string" in context

    answer, rationale = layout_check(context)
    return CheckResult(
        answer=answer,
        aspect="layout check",
        rationale=rationale,
    )


def scale_and_ticks_check_eva(context, query) -> CheckResult:
    assert "base64" in context and "encoding" in context and "chart" in context

    answer, rationale = scale_and_ticks_check(context, query, vision_model)
    return CheckResult(
        answer=answer,
        aspect="scale and ticks check",
        rationale=rationale,
    )


def readability_evaluate(context, query: str) -> list[CheckResult]:
    results = []
    # if self.webdriver_path:
    #     layout_result = self.layout_check(context)
    #     if layout_result.answer is not None:
    #         results.append(layout_result)

    if vision_model:
        context["base64"] = convert_svg_to_base64(context["svg_string"])
        scale_and_ticks_result = scale_and_ticks_check_eva(context, query)
        if scale_and_ticks_result.answer is not None:
            results.append(scale_and_ticks_result)

        aspect_format = {
            "layout check": "Overflow/Overlap",
            "scale and ticks check": "Scale/Ticks",
        }
        reviews = [{
            "aspect": aspect_format[result.aspect],
            "content": result.rationale,
        } for result in results]
        context["reviews"] = reviews

        answer, rationale = readability_check(context, query, vision_model)
        if answer is not None:
            readability_result = CheckResult(
                answer=answer,
                aspect="readability check",
                rationale=rationale,
            )
            results.append(readability_result)

    return results


def convert_svg_to_base64(svg_string):
    png_string = cairosvg.svg2png(bytestring=svg_string)
    base64_encoded = base64.b64encode(png_string).decode("utf-8")
    return f"data:image/png;base64,{base64_encoded}"


def execute_to_svg(code):
    global_env = {"svg_string": None, "show_svg": show_svg, "svg_name": None}

    original_show = plt.show

    code += "\nsvg_string = show_svg(plt,svg_name)"
    # print(code)
    try:

        def dummy_show(*args, **kwargs):
            pass

        plt.show = dummy_show
        exec(code, global_env)
        svg_string = global_env["svg_string"]
        return ChartExecutionResult(status=True, svg_string=svg_string)
    except Exception as exception_error:
        exception_info = traceback.format_exception_only(
            type(exception_error), exception_error)
        return ChartExecutionResult(status=False, error_msg=exception_info)

    finally:

        plt.show = original_show
