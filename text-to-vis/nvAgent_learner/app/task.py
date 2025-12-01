from dataclasses import dataclass, field
from typing import Optional, Any, Dict


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
    difficulty: Optional[str] = field(init=False, default=None)

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
        self.difficulty = task_data.get("difficulty")


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
