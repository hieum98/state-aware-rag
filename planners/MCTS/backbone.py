"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

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

    # @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

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


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, verbose=False):
        self.Q: Dict[Node, float] = defaultdict(float)  # Total reward for each node
        self.N: Dict[Node, int] = defaultdict(int) # Number of visits for each node
        self.children: Dict[Node, List[Node]] = dict() # Children of each node
        self.exploration_weight = exploration_weight # Weight of exploration vs exploitation
        self.verbose = verbose # If True, print debug information

    def choose(self, node: Node):
        """
        Choose the best child of the given node using UCT (Upper Confidence Bound for Trees).
        """
        if node.is_terminal():
            raise ValueError("Cannot choose a child of a terminal node")

        # if node is new, choose a random child
        if node not in self.children:
            return node.find_random_child()
        
        # if node is not new, i.e., in the tree, choose the child with the highest reward
        def score(node):
            if self.N[node] == 0:
                return float('-inf') # If the node has never been visited, return negative infinity to avoid unexplored nodes
            return self.Q[node] / self.N[node] # Average reward of the node
        
        return max(self.children[node], key=score) # Return the child with the highest reward score
    
    def _uct_select(self, node: Node):
        """
        Select a node to go using UCT (Upper Confidence Bound for Trees), balancing exploration and exploitation.
        """
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
    
    def _select(self, node: Node):
        """
        Select a node to expand using UCT (Upper Confidence Bound for Trees).
        The selection process is as follows:
        1. If the node does not have children (terminal node) or the node itself is unexplored, return the node itself.
        2. If the node has children, but not all of them are explored, select a random unexplored child.
        3. If the node has children and all children have been explored, select one child using UCT.
        The selected node is then added to the path. The path is a list of nodes from the root to the selected node.
        """
        path = []
        while True:
            path.append(node)
            # 1. a node does not have children, then select the node itself
            # if the node is unexplored or terminal, return the path
            if node not in self.children or not self.children[node]:
                return path
            # 2. a node has children, but not all of them are explored, select a random unexplored child
            # unexplored nodes are those that are the children of the current node but not explored yet, i.e., hasn't been indexed in the tree
            unexplored = self.children[node] - self.children.keys() 
            if unexplored:
                node = random.choice(list(unexplored)) # Choose a random unexplored node to expand
                path.append(node)
                return path
            # 3. a node has children and all children have been explored, then select one child and go to the next layer
            # if the node is fully explored, select the child with UCT
            node = self._uct_select(node)

    def _expand(self, node: Node):
        """
        Expand the given node by adding its children to the tree.
        Update the tree with the new children of unexplored nodes.
        """
        if node in self.children:
            return # Already expanded
        if node.is_terminal():
            return # Terminal node, no children to expand
        self.children[node] = node.find_children()
        if self.verbose:
            print(f"Expanding node:")
            node.print_node()
            print(f"Found {len(self.children[node])} children:")
            for i, child in enumerate(self.children[node]):
                print("*" * 10)
                print(f"Child {i}:")
                child.print_node()
    
    def _simulate(self, node: Node):
        """
        Simulate a random game from the given node to a terminal state.
        Return the reward of the terminal state.
        """
        while True:
            if node.is_terminal():
                return node
            # if the node is not terminal, select a random child and go to the next layer
            node = node.find_random_child()
    
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
    
    def do_rollout(self, node: Node):
        """
        Perform a rollout from the given node to a terminal state and update the tree's nodes reward and visit counts by using backpropagation.
        """
        path = self._select(node) # Select a path to expand
        if self.verbose:
            print("Selected path for rollout:")
            for i, n in enumerate(path):
                print(f"Step {i}: ")
                n.print_node()
        leaf = path[-1] # The last node in the path is the leaf node
        self._expand(leaf) # Expand the the tree with the children of the leaf node
        simulated_node = self._simulate(leaf) # Simulate a random game from the leaf node to a terminal state
        if self.verbose:
            print(f"Simulated node:")
            simulated_node.print_node()
        reward = simulated_node.reward()
        if self.verbose:
            print(f"Reward for the simulated node: {reward}")
        self._backpropagate(path, reward)
        if self.verbose:
            print("Backpropagating the reward:")
            for i, n in enumerate(path):
                print(f"Step {i}:")
                n.print_node()
                print(f"Reward: {self.Q[n]}, Visits: {self.N[n]}")
        return simulated_node
    
