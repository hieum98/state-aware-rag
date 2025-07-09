import datasets
from typing import Any, Dict, List, Optional, Union

from agents.roles.evaluator import Evaluator
from agents.roles.extractor import Extractor
from agents.roles.generator import Generator
from agents.retriever_agents import RetrieverAgent
from planners.MCTS.utils import search


def generate_answer(
        question: Union[str, List[str]],
        generator: Generator,
        evaluator: Evaluator,
        extractor: Extractor,
        retriever: RetrieverAgent,
        # Optional parameters
        question_id: Optional[str] = None,
        goden_answer: Optional[Union[str, List[str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
    final_answer, solutions = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question=question,
        question_id=question_id,
        golden_answer=goden_answer,
        # MCTS parameters
        max_depth=config.get("max_depth", 15) if config else 15,
        num_rollouts=config.get("num_rollouts", 100) if config else 100,
        use_golden_answer=config.get("use_golden_answer", False) if config else False,
        save_tree=config.get("save_tree", False) if config else False,
        save_dir=config.get("save_dir", "mcts_data") if config else None
    )
    return {
        "final_answer": final_answer,
        "solutions": solutions
    }


