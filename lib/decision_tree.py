import math
from anytree import NodeMixin, RenderTree
from anytree.exporter import DotExporter
import subprocess


class DecisionTreeNode(NodeMixin):
    """
    Class representing a node in the decision tree.
    Attributes:
    - name: Name of the node (attribute or decision)
    - attribute_index: Index of the attribute if it's a decision node
    - decision: Decision value if it's a leaf node
    - parent: Parent node
    """

    def __init__(
        self,
        name: str,
        attribute_index: int = None,
        decision: str = None,
        parent=None,
    ):
        super().__init__()
        self.name = name
        self.attribute_index = attribute_index
        self.decision = decision
        self.parent = parent


class DecisionTree:
    """Class representing a decision tree.
    Methods:
    - load_data(file_name): Load data from a file and preprocess it
    - print_tree: Print the decision tree
    - save_tree_to_file: Save the decision tree to a txt file
    - export_tree_to_graphviz: Export the decision tree to a Graphviz DOT file
    """

    def __init__(self) -> None:
        self.tree_root = None
        self.data = []
        self.attributes = []
        self.decisions = []
        self.unique_attributes_values = {}
        self.value_occurrences = {}
        self.decision_counts = {}
        self.attribute_decision_counts = {}

    def load_data(self, file_name: str) -> None:
        """Load data from a file and preprocess it."""
        with open(file_name, "r") as file:
            self.data = [line.strip().split(",") for line in file.readlines()]
        self.attributes = [row[:-1] for row in self.data]
        self.decisions = [row[-1] for row in self.data]
        self._count_unique_attributes_values()
        self._count_value_occurrences()
        self._calculate_decision_counts()
        self._calculate_attribute_decision_counts()
        self.tree_root = self._build_tree()

    def print_tree(self) -> None:
        if self.tree_root:
            for pre, _, node in RenderTree(self.tree_root):
                print(f"{pre}{node.name}")
        else:
            print("The tree is empty.")

    def save_tree_to_file(self, filename: str = "output/decision_tree.txt") -> None:
        if self.tree_root:
            with open(filename, "w", encoding="utf-8") as f:
                for pre, _, node in RenderTree(self.tree_root):
                    f.write(f"{pre}{node.name}\n")
            print(f"Decision tree saved to {filename}")
        else:
            print("The tree is empty. Cannot save to file.")

    def export_tree_to_graphviz(
        self,
        dot_filename: str = "output/decision_tree.dot",
        svg_filename: str = "output/decision_tree.svg",
    ) -> None:
        """Export the decision tree to a Graphviz DOT file."""
        if self.tree_root:
            DotExporter(
                self.tree_root,
                graph="digraph",
                options=["ranksep=7; nodesep=1; overlap=false"],
            ).to_dotfile(dot_filename)
            print(f"Decision tree exported to {dot_filename}")
            try:
                subprocess.run(
                    ["dot", "-Tsvg", "-Kdot", dot_filename, "-o", svg_filename],
                    check=True,
                )
                print(f"Decision tree rendered to {svg_filename}")
            except FileNotFoundError:
                print("Graphviz is not installed or 'dot' command is not in PATH.")
            except subprocess.CalledProcessError as e:
                print(f"Error rendering the DOT file: {e}")
        else:
            print("The tree is empty. Cannot export.")

    def _build_tree(
        self,
        data_indices: list = None,
        used_attributes: set = None,
        parent: DecisionTreeNode = None,
    ) -> DecisionTreeNode:
        if data_indices is None:
            data_indices = list(range(len(self.decisions)))
        if used_attributes is None:
            used_attributes = set()

        decisions = [self.decisions[i] for i in data_indices]
        if len(set(decisions)) == 1:
            return DecisionTreeNode(
                name=f"Decision: {decisions[0]}", decision=decisions[0], parent=parent
            )

        best_attr, best_gain = None, -1
        for attr in self.unique_attributes_values:
            if attr not in used_attributes:
                gain = self.calculate_gain_ratio(attr)
                if gain > best_gain:
                    best_attr, best_gain = attr, gain

        if best_attr is None:
            majority = max(set(decisions), key=decisions.count)
            return DecisionTreeNode(
                name=f"Decision: {majority}", decision=majority, parent=parent
            )

        node = DecisionTreeNode(
            name=f"Attribute a{best_attr + 1}", attribute_index=best_attr, parent=parent
        )

        for value in self.value_occurrences[best_attr]:
            subset_indices = [
                i for i in data_indices if self.attributes[i][best_attr] == value
            ]
            if subset_indices:
                child = self._build_tree(
                    subset_indices, used_attributes | {best_attr}, parent=node
                )
                child.name = f"{value} ➔ {child.name}"
            else:
                majority = max(set(decisions), key=decisions.count)
                DecisionTreeNode(
                    name=f"{value} ➔ Decision: {majority}",
                    decision=majority,
                    parent=node,
                )

        return node

    def calculate_entropy(self) -> float:
        """Calculate entropy of the entire dataset"""
        return self._calculate_entropy_from_counts(self.decision_counts)

    def calculate_conditional_entropy(self, attribute_index: int) -> float:
        """Calculate the conditional entropy for a specific attribute."""
        conditional_entropy = 0.0
        total = len(self.decisions)
        for value, decision_counts in self.attribute_decision_counts[
            attribute_index
        ].items():
            value_probability = sum(decision_counts.values()) / total
            value_entropy = self._calculate_entropy_from_counts(decision_counts)
            conditional_entropy += value_probability * value_entropy
        return conditional_entropy

    def calculate_information_gain(self, attribute_index: int) -> float:
        """Calculate information gain for an attribute"""
        return self.calculate_entropy() - self.calculate_conditional_entropy(
            attribute_index
        )

    def calculate_split_information(self, attribute_index: int) -> float:
        """Calculate split information for an attribute (used in gain ratio)"""
        value_counts = self.value_occurrences[attribute_index]
        return self._calculate_entropy_from_counts(value_counts)

    def calculate_gain_ratio(self, attribute_index: int) -> float:
        """Calculate gain ratio for an attribute. Used for normalization of information gain."""
        information_gain = self.calculate_information_gain(attribute_index)
        split_info = self.calculate_split_information(attribute_index)

        if split_info == 0:
            return 0
        return information_gain / split_info

    def _calculate_entropy_from_counts(self, decision_counts: dict) -> float:
        """Calculate entropy directly from decision counts."""
        total = sum(decision_counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in decision_counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        return entropy

    def _count_unique_attributes_values(self) -> None:
        """Count the number of unique values for each attribute."""
        unique_values = {}
        for i in range(len(self.attributes[0])):
            unique_values[i] = set(row[i] for row in self.attributes)
        self.unique_attributes_values = {
            key: len(values) for key, values in unique_values.items()
        }

    def _count_value_occurrences(self) -> None:
        """Count occurrences of each value for every attribute."""
        self.value_occurrences = {}
        for i in range(len(self.attributes[0])):
            self.value_occurrences[i] = {}
            for row in self.attributes:
                value = row[i]
                self.value_occurrences[i][value] = (
                    self.value_occurrences[i].get(value, 0) + 1
                )

    def _calculate_decision_counts(self) -> None:
        """Count occurrences of each decision."""
        self.decision_counts = {}
        for decision in self.decisions:
            self.decision_counts[decision] = self.decision_counts.get(decision, 0) + 1

    def _calculate_attribute_decision_counts(self) -> None:
        """Count decisions for each value of every attribute."""
        self.attribute_decision_counts = {}
        for i in range(len(self.attributes[0])):
            self.attribute_decision_counts[i] = {}
            for value, decision in zip(
                [row[i] for row in self.attributes], self.decisions
            ):
                if value not in self.attribute_decision_counts[i]:
                    self.attribute_decision_counts[i][value] = {}
                self.attribute_decision_counts[i][value][decision] = (
                    self.attribute_decision_counts[i][value].get(decision, 0) + 1
                )
