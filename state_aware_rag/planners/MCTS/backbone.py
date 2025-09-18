"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random
import logging

from state_aware_rag.planners.reasoning_node import ReasoningNode as Node

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, verbose=False):
        self.Q: Dict[Node, float] = defaultdict(float)  # Total reward for each node
        self.N: Dict[Node, int] = defaultdict(int) # Number of visits for each node
        self.children: Dict[Node, List[Node]] = dict() # Children of each node
        self.exploration_weight = exploration_weight # Weight of exploration vs exploitation
        
        self.explored_nodes = set() # Set of explored nodes
    
    def _uct_select(self, node: Node):
        """
        Select a node to go using UCT (Upper Confidence Bound for Trees), balancing exploration and exploitation.
        """
        # All children of node should already be expanded:
        assert all(n in self.explored_nodes for n in self.children[node]), "All children of the node should be explored before UCT selection"
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            if self.N[n] == 0:
                return float('inf')
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
    
    def _select(self, node: Node) -> List[Node]:
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
            if node not in self.children.keys():
                return path
            
            # 2. a node has children, but not all of them are explored, select a random unexplored child
            # unexplored nodes are those that are the children of the current node but not explored yet, i.e., hasn't been indexed in the tree
            unexplored = [n for n in self.children[node] if n not in self.explored_nodes]
            if unexplored:
                node = random.choice(unexplored) # Choose a random unexplored node to expand
                path.append(node)
                return path
            
            # 3. a node has children and all children have been explored, then select one child and go to the next layer
            # if the node is fully explored, select the child with UCT
            node = self._uct_select(node)

    def _expand(self, node: Node, rollout_id=None):
        """
        Expand the given node by adding its children to the tree.
        Update the tree with the new children of unexplored nodes.
        """
        if node in self.explored_nodes:
            return # Already expanded
        if node.is_terminal():
            self.explored_nodes.add(node)  # Mark the node as explored
            return # Terminal node, no children to expand
        if node is None:
            return # None node, no children to expand
        
        children = node.find_children(rollout_id) # Find the children of the node
        if children:
            self.children[node] = children # Add the children to the tree
            logger.debug(f"Expanding node: {node}. Found {len(self.children[node])} children.")
    
    def _simulate(self, node: Node, rollout_id=None) -> List[Node]:
        """
        Simulate a random game from the given node to a terminal state.
        Return the reward of the terminal state.
        """
        path = []
        current_node = node
        while True:
            if current_node.is_terminal():
                self.explored_nodes.add(current_node)  # Mark the node as explored
                return path
            if current_node is None:
                return path
            
            if current_node not in self.children.keys():
                children = current_node.find_children(rollout_id)
                if children:
                    self.children[current_node] = children # Expand the node if it has no children
            
            # Handle empty child lists gracefully
            if not self.children[current_node]:
                self.explored_nodes.add(current_node)
                return path
            
            current_node = random.choice(self.children[current_node]) # Choose a random child to simulate
            path.append(current_node) # Add the current node to the path
    
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            self.explored_nodes.add(node) # Mark the node as explored
    
    def do_rollout(self, node: Node, rollout_id=None):
        """
        Perform a rollout from the given node to a terminal state and update the tree's nodes reward and visit counts by using backpropagation.
        """
        path = self._select(node) # Select a path to expand
        leaf = path[-1] # The last node in the path is the leaf node
        
        self._expand(leaf, rollout_id=rollout_id) # Expand the the tree with the children of the leaf node
        simulated_path = self._simulate(leaf, rollout_id=rollout_id) # Simulate a random game from the leaf node to a terminal state
        if simulated_path:
            simulated_node = simulated_path[-1] # The last node in the simulated path is the terminal node
        else:
            simulated_node = path[-1] # If the simulated path is empty, use the leaf node as the simulated node
                
        # Guard reward computation for non-leaf nodes
        reward = simulated_node.reward() if hasattr(simulated_node, 'is_valid_leaf') and simulated_node.is_valid_leaf() else 0.0
        
        self._backpropagate(path + simulated_path, reward)
        
        return simulated_node
