import asyncio
import omegaconf
import pytest
from state_aware_rag.planners.reasoning_node import ReasoningNode, ExtractorAgent, NodeType

def test_reasoning_node(ray_cluster):
    # Load generator config and make it safe for tests (no global rate limiter actor)
    cfg = omegaconf.OmegaConf.load("configs/extractor.yaml")
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["enable_global_rate_limit"] = True
    cfg_dict["num_workers"] = 4
    cfg_dict["timeout"] = 30
    cfg_dict["verbose"] = True

    agent = ExtractorAgent(config=cfg_dict)

    node = ReasoningNode(
        node_type=NodeType.USER_QUESTION,
        extractor=agent,
    )
    children_node = ReasoningNode(
        node_type=NodeType.SUB_QA_NODE,
        parent=node,
        extractor=node.extractor,
    )