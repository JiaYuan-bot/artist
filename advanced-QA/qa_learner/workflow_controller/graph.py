from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx


class NodeType(Enum):
    """Types of nodes in the workflow graph"""
    START = "start"
    END = "end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    CODE_EXECUTION = "code_execution"
    DATA_RETRIEVAL = "data_retrieval"
    FUNCTION_CALL = "function_call"


class FunctionName(Enum):
    """Function names for workflow nodes"""
    CANDIDATE_GENERATION = "candidate_generation"
    COLUMN_FILTERING = "column_filtering"
    COLUMN_SELECTION = "column_selection"
    CONTEXT_RETRIEVAL = "context_retrieval"
    ENTITY_RETRIEVAL = "entity_retrieval"
    EVALUATION = "evaluation"
    KEYWORD_EXTRACTION = "keyword_extraction"
    REVISION = "revision"
    TABLE_SELECTION = "table_selection"

    PREPROCESS = "preprocess"
    PROCESSOR = "processor"
    COMPOSER = "composer"
    VALIDATOR = "validator"
    NVEVALUATION = "nv_evaluation"

    QUESTION_TO_QUERY = "question_to_query"
    RETRIEVAL1 = "retrieval1"
    CONTEXT_QUESTION_TO_QUERY = "context_question_to_query"
    RETRIEVAL2 = "retrieval2"
    RERANK = "rerank"
    CONTEXT_QUESTION_TO_ANSWER = "context_question_to_answer"
    QA_EVALUATION = "qa_evaluation"


@dataclass
class NodeConfig:
    """Configuration for a node"""
    model: str = ""
    temperature: float = 0.0
    template_name: str = ""
    parser_name: str = ""
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Make NodeConfig hashable for use in dictionaries"""
        return hash((self.model, self.temperature, self.template_name,
                     tuple(sorted(self.custom_params.items()))))


@dataclass
class Node:
    """Represents a node in the workflow graph"""
    id: str
    name: str
    node_type: NodeType
    function_name: Optional[FunctionName] = None
    config: NodeConfig = field(default_factory=NodeConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id


@dataclass
class Edge:
    """Represents an edge in the workflow graph"""
    source_id: str
    target_id: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowGraph:
    """Workflow graph with START and END nodes"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}

        # Add START and END nodes by default
        self._init_start_end_nodes()

    def _init_start_end_nodes(self):
        """Initialize special START and END nodes"""
        start_node = Node(id="START",
                          name="Start",
                          node_type=NodeType.START,
                          metadata={"special": True})
        end_node = Node(id="END",
                        name="End",
                        node_type=NodeType.END,
                        metadata={"special": True})

        self.nodes["START"] = start_node
        self.nodes["END"] = end_node
        self.graph.add_node("START", node=start_node)
        self.graph.add_node("END", node=end_node)

    def add_node(self, node: Node):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node=node)

    def add_edge(self, edge: Edge):
        """Add an edge to the graph"""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(
                f"Both source and target nodes must exist in the graph,{edge.source_id }, {edge.target_id}"
            )

        self.graph.add_edge(edge.source_id,
                            edge.target_id,
                            weight=edge.weight,
                            metadata=edge.metadata)

    def get_all_paths(self,
                      source_id: str = "START",
                      target_id: str = "END") -> List[List[str]]:
        """Get all possible paths from source to target (default: START to END)"""
        return list(nx.all_simple_paths(self.graph, source_id, target_id))

    def get_node_successors(self,
                            node_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all successors of a node with edge data"""
        successors = []
        for successor_id in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, successor_id)
            successors.append((successor_id, edge_data))
        return successors

    def get_node_predecessors(
            self, node_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all predecessors of a node with edge data"""
        predecessors = []
        for predecessor_id in self.graph.predecessors(node_id):
            edge_data = self.graph.get_edge_data(predecessor_id, node_id)
            predecessors.append((predecessor_id, edge_data))
        return predecessors

    def has_path(self, source_id: str, target_id: str) -> bool:
        """Check if a path exists between two nodes"""
        return nx.has_path(self.graph, source_id, target_id)

    def get_shortest_path(self,
                          source_id: str = "START",
                          target_id: str = "END") -> List[str]:
        """Get the shortest path from source to target"""
        try:
            return nx.shortest_path(self.graph,
                                    source_id,
                                    target_id,
                                    weight='weight')
        except nx.NetworkXNoPath:
            return []

    def visualize(self, save_path: Optional[str] = None):
        """Visualize the workflow graph (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph, k=2, iterations=50)

            # Separate nodes by type
            start_end_nodes = [
                n for n in self.graph.nodes() if n in ["START", "END"]
            ]
            other_nodes = [
                n for n in self.graph.nodes() if n not in ["START", "END"]
            ]

            # Draw nodes
            nx.draw_networkx_nodes(self.graph,
                                   pos,
                                   nodelist=start_end_nodes,
                                   node_color='lightgreen',
                                   node_size=1500,
                                   node_shape='s')
            nx.draw_networkx_nodes(self.graph,
                                   pos,
                                   nodelist=other_nodes,
                                   node_color='lightblue',
                                   node_size=1000)

            # Draw edges
            nx.draw_networkx_edges(self.graph,
                                   pos,
                                   edge_color='gray',
                                   arrows=True,
                                   arrowsize=20,
                                   width=2,
                                   alpha=0.7)

            # Draw labels
            labels = {
                node_id: self.nodes[node_id].name
                for node_id in self.graph.nodes()
            }
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)

            plt.title("Workflow Graph")
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available. Cannot visualize graph.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the workflow graph to a dictionary"""
        return {
            "nodes": {
                node_id: {
                    "id":
                    node.id,
                    "name":
                    node.name,
                    "node_type":
                    node.node_type.value,
                    "function_name":
                    node.function_name.value if node.function_name else None,
                    "config": {
                        "model": node.config.model,
                        "temperature": node.config.temperature,
                        "template_name": node.config.template_name,
                        "custom_params": node.config.custom_params
                    },
                    "metadata":
                    node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [{
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
                "metadata": data.get("metadata", {})
            } for u, v, data in self.graph.edges(data=True)]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowGraph':
        """Create a workflow graph from a dictionary"""
        graph = cls()

        # Add nodes (skip START and END as they're already added)
        for node_id, node_data in data["nodes"].items():
            if node_id not in ["START", "END"]:
                config = NodeConfig(
                    model=node_data["config"]["model"],
                    temperature=node_data["config"]["temperature"],
                    template_name=node_data["config"]["template_name"],
                    custom_params=node_data["config"]["custom_params"])
                node = Node(
                    id=node_data["id"],
                    name=node_data["name"],
                    node_type=NodeType(node_data["node_type"]),
                    function_name=FunctionName(node_data["function_name"])
                    if node_data["function_name"] else None,
                    config=config,
                    metadata=node_data["metadata"])
                graph.add_node(node)

        # Add edges
        for edge_data in data["edges"]:
            edge = Edge(source_id=edge_data["source"],
                        target_id=edge_data["target"],
                        weight=edge_data["weight"],
                        metadata=edge_data["metadata"])
            graph.add_edge(edge)

        return graph


def create_example_qa_workflow_default(
        model="gemini-2.5-pro") -> WorkflowGraph:
    """Create an example workflow for text-to-SQL"""
    graph = WorkflowGraph()

    question_to_query_large = Node(
        id="question_to_query_large",
        name="question_to_query_large",
        node_type=NodeType.LLM_CALL,
        function_name=FunctionName.QUESTION_TO_QUERY,
        config=NodeConfig(model=model))

    retrieval1_10 = Node(id="retrieval1_10",
                         name="retrieval1_10",
                         node_type=NodeType.TOOL_CALL,
                         function_name=FunctionName.RETRIEVAL1,
                         config=NodeConfig(model=model))

    context_question_to_query_large = Node(
        id="context_question_to_query_large",
        name="context_question_to_query_large",
        node_type=NodeType.LLM_CALL,
        function_name=FunctionName.CONTEXT_QUESTION_TO_QUERY,
        config=NodeConfig(model=model))

    retrieval2_10 = Node(id="retrieval2_10",
                         name="retrieval2_10",
                         node_type=NodeType.TOOL_CALL,
                         function_name=FunctionName.RETRIEVAL2,
                         config=NodeConfig(model=model))

    rerank_large = Node(id="rerank_large",
                        name="rerank_large",
                        node_type=NodeType.LLM_CALL,
                        function_name=FunctionName.RERANK,
                        config=NodeConfig(model=model))

    context_question_to_answer_large_cot = Node(
        id="context_question_to_answer_large_cot",
        name="context_question_to_answer_large_cot",
        node_type=NodeType.LLM_CALL,
        function_name=FunctionName.CONTEXT_QUESTION_TO_ANSWER,
        config=NodeConfig(
            model=model,
            template_name=
            "context_question_to_answer_cot_template",  #context_question_to_answer_cot_template
        ))

    qa_evaluation = Node(id="qa_evaluation",
                         name="qa_evaluation",
                         node_type=NodeType.CODE_EXECUTION,
                         function_name=FunctionName.QA_EVALUATION,
                         config=NodeConfig())

    # Add all nodes
    for node in [
            question_to_query_large, retrieval1_10,
            context_question_to_query_large, retrieval2_10, rerank_large,
            context_question_to_answer_large_cot, qa_evaluation
    ]:
        graph.add_node(node)

    # Connect START to entry nodes
    graph.add_edge(Edge("START", "question_to_query_large", weight=0.5))

    graph.add_edge(Edge("question_to_query_large", "retrieval1_10",
                        weight=0.5))

    graph.add_edge(
        Edge("retrieval1_10", "context_question_to_query_large", weight=0.5))

    graph.add_edge(
        Edge("context_question_to_query_large", "retrieval2_10", weight=0.5))

    graph.add_edge(Edge("retrieval2_10", "rerank_large", weight=0.5))

    graph.add_edge(
        Edge("rerank_large",
             "context_question_to_answer_large_cot",
             weight=0.5))

    graph.add_edge(
        Edge("context_question_to_answer_large_cot",
             "qa_evaluation",
             weight=0.5))

    graph.add_edge(Edge("qa_evaluation", "END", weight=1.0))

    return graph


if __name__ == "__main__":
    # Example usage
    example_graph = create_example_qa_workflow_default()
    paths = example_graph.get_all_paths()
    print(len(paths))
