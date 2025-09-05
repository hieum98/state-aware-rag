import asyncio
import omegaconf
import pytest
from state_aware_rag.planners.reasoning_node import *


def test_user_question_node():
    generator_config = omegaconf.OmegaConf.load("configs/generator.yaml")
    generator_config = omegaconf.OmegaConf.to_container(generator_config, resolve=True)
    retriever_config = omegaconf.OmegaConf.load("configs/retriever.yaml")
    retriever_config = omegaconf.OmegaConf.to_container(retriever_config, resolve=True)
    extractor_config = omegaconf.OmegaConf.load("configs/extractor.yaml")
    extractor_config = omegaconf.OmegaConf.to_container(extractor_config, resolve=True)
    evaluator_config = omegaconf.OmegaConf.load("configs/evaluator.yaml")
    evaluator_config = omegaconf.OmegaConf.to_container(evaluator_config, resolve=True)
    node_config = omegaconf.OmegaConf.load("configs/mcts.yaml")

    generator = GeneratorAgent(config=generator_config)
    retriever = RetrievalAgent(config=retriever_config)
    extractor = ExtractorAgent(config=extractor_config)
    evaluator = EvaluatorAgent(config=evaluator_config)

    node = ReasoningNode(
        node_type=NodeType.USER_QUESTION,
        parent=None,
        # Node components
        generator=generator,
        retriever=retriever,
        extractor=extractor,
        evaluator=evaluator,
        # Node data
        # question: Optional[str] = None,
        # answer: Optional[str] = None,
        # reasoning: Optional[str] = None,
        # confidence: Optional[float] = None,
        # memory: Optional[List[str]] = None,
        # Options
        is_cot=node_config.get("is_cot", False),
        max_depth=node_config.get("max_depth", 5),
        golden_answer="The New York Times was founded earlier, in 1851",
        user_question="which magazine was founded earlier, The New York Times or Wall Street Journal?",
        question_id="test_001",
        top_k=node_config.get("top_k", 3),
    )

    children = asyncio.run(node.generate_children())
    for child in children:
        print("Child Node:")
        print(child)

def test_sub_qa_node():
    generator_config = omegaconf.OmegaConf.load("configs/generator.yaml")
    generator_config = omegaconf.OmegaConf.to_container(generator_config, resolve=True)
    retriever_config = omegaconf.OmegaConf.load("configs/retriever.yaml")
    retriever_config = omegaconf.OmegaConf.to_container(retriever_config, resolve=True)
    extractor_config = omegaconf.OmegaConf.load("configs/extractor.yaml")
    extractor_config = omegaconf.OmegaConf.to_container(extractor_config, resolve=True)
    evaluator_config = omegaconf.OmegaConf.load("configs/evaluator.yaml")
    evaluator_config = omegaconf.OmegaConf.to_container(evaluator_config, resolve=True)
    node_config = omegaconf.OmegaConf.load("configs/mcts.yaml")

    generator = GeneratorAgent(config=generator_config)
    retriever = RetrievalAgent(config=retriever_config)
    extractor = ExtractorAgent(config=extractor_config)
    evaluator = EvaluatorAgent(config=evaluator_config)

    node = ReasoningNode(
        node_type=NodeType.SUB_QA_NODE,
        parent=None,
        # Node components
        generator=generator,
        retriever=retriever,
        extractor=extractor,
        evaluator=evaluator,
        # Node data
        question="When was The New York Times founded?",
        answer="1851",
        reasoning="The New York Times was founded in 1851",
        confidence=1.0,
        memory=["The New York Times was founded in 1851", "Name of the magazine is The New York Times"],
        # Options
        is_cot=node_config.get("is_cot", False),
        max_depth=node_config.get("max_depth", 5),
        golden_answer="The New York Times was founded earlier, in 1851",
        user_question="which magazine was founded earlier, The New York Times or Wall Street Journal?",
        question_id="test_001",
        top_k=node_config.get("top_k", 3),
    )

    children = asyncio.run(node.generate_children())
    for child in children:
        print("Child Node:")
        print(child)
        
    

