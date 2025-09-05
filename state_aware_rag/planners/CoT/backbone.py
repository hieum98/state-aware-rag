import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random

from state_aware_rag.planners.reasoning_node import ReasoningNode as Node

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


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
            logger.warning(f"No children found for node type: {current_node.print_node()}")
            return path
        if all([child.is_valid_leaf() for child in children]):
            current_node = random.choice(list(children))
        else:
            assert len(children) == 1, "Expected exactly one child node if children are not valid leaves, getting: {} children".format(len(children))
            current_node = children[0]
        if current_node is None:
            logger.warning("Current node is None, stopping rollout.")
            return path
        path.append(current_node)
                
        
