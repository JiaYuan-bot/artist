from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import json


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

        # self.difficulty = task_data.get("difficulty")


from dataclasses import dataclass, field
from typing import Dict, Any, List
from datasets import load_dataset


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


# def load_train_set() -> List[QATask]:
#     """
#     Loads the HotpotQA 'train' split and converts it into a list of QATask objects
#     with sequential integer IDs.

#     Returns:
#         List[QATask]: A list of QATask instances.
#     """
#     print("Loading HotpotQA training set...")
#     dataset = load_dataset("hotpot_qa", 'fullwiki', split='train')

#     # Use enumerate to generate an index (idx) for each task
#     train_tasks = [
#         QATask(idx, task_data) for idx, task_data in enumerate(dataset)
#     ]
#     print(f"Loaded {len(train_tasks)} tasks from the training set.")
#     return train_tasks

# import random
# from typing import List, Optional

# def load_test_set(num_samples: Optional[int] = None,
#               seed: int = 42) -> List[QATask]:
# """
# Loads the HotpotQA 'validation' split and converts it into a list of QATask objects
# with sequential integer IDs.

# Args:
#     num_samples: Number of samples to randomly select. If None, load all data.
#     seed: Random seed for reproducibility (default: 42).

# Returns:
#     List[QATask]: A list of QATask instances.
# """
# print("Loading HotpotQA validation (test) set...")
# dataset = load_dataset("hotpot_qa", 'fullwiki', split='validation')

# print(f"Total available tasks: {len(dataset)}")

# # If num_samples is specified and less than dataset size, randomly sample
# if num_samples is not None and num_samples < len(dataset):
#     random.seed(seed)
#     # Randomly select indices
#     sampled_indices = random.sample(range(len(dataset)), num_samples)
#     sampled_indices.sort()  # Optional: keep original order

#     # Create tasks only for sampled indices
#     test_tasks = [
#         QATask(idx, dataset[i]) for idx, i in enumerate(sampled_indices)
#     ]
#     print(
#         f"Randomly sampled {len(test_tasks)} tasks from the validation set."
#     )
# else:
#     # Load all data
#     test_tasks = [
#         QATask(idx, task_data) for idx, task_data in enumerate(dataset)
#     ]
#     print(f"Loaded all {len(test_tasks)} tasks from the validation set.")

# return test_tasks

# --- Example of how to use the functions ---
if __name__ == "__main__":
    training_data = load_test_dataset(
        path="/home/jiayuan/nl2sql/MultiHop-RAG/dataset/qa_test.json")
    print(training_data[0])
