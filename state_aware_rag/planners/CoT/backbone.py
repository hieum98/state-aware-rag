from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random


class Node(ABC):
    """
    A node in the MCTS tree. This is an abstract base class that should be subclassed to implement specific game logic.
    """
    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
    
    @abstractmethod
    def print_node(self):
        "Print the node in a human-readable format"
        pass


def do_rollout(node: Node):
    """
    Perform a rollout from the given node.
    """
    path = []
    current_node = node
    while True:
        if current_node.is_terminal():
            return path
        children = current_node.find_children()
        if len(children) == 0:
            print(f"No children found for node type: {current_node.print_node()}")
            return path
        if all([child.is_valid_leaf() for child in children]):
            current_node = random.choice(list(children))
        else:
            assert len(children) == 1, "Expected exactly one child node if children are not valid leaves, getting: {} children".format(len(children))
            current_node = children[0]
        if current_node is None:
            print("Current node is None, returning path")
            return path
        path.append(current_node)
                
        
