import omegaconf

from state_aware_rag.agents.agents import EvaluatorAgent, ExtractorAgent, GeneratorAgent, RetrievalAgent
from state_aware_rag.planners.MCTS.utils import search


def test_mcts():
    generator_config = omegaconf.OmegaConf.load("configs/generator.yaml")
    generator_config = omegaconf.OmegaConf.to_container(generator_config, resolve=True)
    retriever_config = omegaconf.OmegaConf.load("configs/retriever.yaml")
    retriever_config = omegaconf.OmegaConf.to_container(retriever_config, resolve=True)
    extractor_config = omegaconf.OmegaConf.load("configs/extractor.yaml")
    extractor_config = omegaconf.OmegaConf.to_container(extractor_config, resolve=True)
    evaluator_config = omegaconf.OmegaConf.load("configs/evaluator.yaml")
    evaluator_config = omegaconf.OmegaConf.to_container(evaluator_config, resolve=True)
    search_config = omegaconf.OmegaConf.load("configs/mcts.yaml")

    generator = GeneratorAgent(config=generator_config)
    retriever = RetrievalAgent(config=retriever_config)
    extractor = ExtractorAgent(config=extractor_config)
    evaluator = EvaluatorAgent(config=evaluator_config)

    final_answer, final_reasoning = search(
        generator=generator,
        evaluator=evaluator,
        extractor=extractor,
        retriever=retriever,
        # Question components
        user_question="Which magazine was started first Arthur's Magazine or First for Women?",
        question_id="test_001",
        golden_answer="Arthur's Magazine",
        # MCTS parameters
        max_depth=5, # search_config.max_depth,
        num_rollouts=3, # search_config.num_rollouts
        exploration_weight=search_config.exploration_weight,
        use_golden_answer=search_config.use_golden_answer,
        save_tree=search_config.save_tree,
        save_dir=search_config.tree_dir,
        top_k=search_config.top_k,
        verbose=True,
    )
    print("Final Answer:", final_answer)
    print("Final Reasoning:", final_reasoning)


