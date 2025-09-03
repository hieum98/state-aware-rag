import asyncio
import logging
import os
import random
import copy
import json
from enum import Enum, unique
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import tqdm
import pprint
from anytree import NodeMixin

from state_aware_rag.agents.agents import (
    GeneratorAgent, 
    RetrievalAgent,
    ExtractorAgent,
    RetrievalAgent
    )
from state_aware_rag.agents.prompts import (
    decompose_and_answer,
    evaluate,
    extract,
    finalize,
    rephase_question,
    self_correct,
    synthesize
    )
from state_aware_rag.agents.utils import (
    format_reasoning_trace,
    format_memory,
    format_context,
    format_reflection_context,
    convert_confidence_to_score
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGLEVEL", "INFO"))
class NodeType(Enum):
    # Node type for user question, i.e., the root node
    USER_QUESTION = "USER_QUESTION"
    # Node type for final answer of the user question, i.e., the terminal node  
    FINAL_ANSWER = "FINAL_ANSWER"
    # Node type for subQA, i.e., the intermediate node for the sub-question and sub-answer 
    SUB_QA_NODE = "SUBQUESTION" 
    # Node type for rephrase question, i.e., the intermediate node for rephrased question. 
    # This node must be followed by a SUBQUESTION node and be generated from a USER_QUESTION or SUBQUESTION node.
    REPHASED_QUESTION_NODE = "REPHASE_QUESTION"
    # Node type for self-correcting reasoning, i.e., the intermediate node for self-correcting reasoning.
    # This node must be generated from a SUBQUESTION node
    SELF_CORRECTED_NODE = "SELF_CORRECT"
    # Node type for reasoning strengthening, i.e., the intermediate node for reasoning strengthening.  
    # This node must be generated from a SUBQUESTION or SELF_CORRECTED_NODE node
    SYNTHESIS_NODE = "SYNTHESIS" 


class ReasoningNode(NodeMixin):
    def __init__(
            self,
            node_type: NodeType,
            parent: Optional['ReasoningNode'] = None,
            # Node components
            generator: Optional[GeneratorAgent] = None,
            retriever: Optional[RetrievalAgent] = None,
            extractor: Optional[ExtractorAgent] = None,
            evaluator: Optional[GeneratorAgent] = None,
            # Node data
            question: Optional[str] = None,
            answer: Optional[str] = None,
            reasoning: Optional[str] = None,
            confidence: Optional[float] = None,
            memory: Optional[List[str]] = None,
            # Options
            max_depth: int = 15,
            golden_answer: Optional[Union[str, List[str]]] = None,
            user_question: Optional[str] = None,
            question_id: Optional[str] = None,
            top_k: int = None,
            **kwargs
            ):
        super().__init__()

        self.node_config = {
            # Node args
            "max_depth": max_depth,
            "golden_answer": golden_answer,
            "user_question": user_question,
            "question_id": question_id,
            "top_k": top_k,
            # Node agents
            "generator": generator,
            "retriever": retriever,
            "evaluator": evaluator,
            "extractor": extractor,
        }
        # Node topology
        self.node_type = node_type
        self.parent = parent
        self.children: List['ReasoningNode'] = []
        self.max_depth = max_depth
        # Node agents
        self.generator = generator
        self.retriever = retriever
        self.extractor = extractor
        self.evaluator = evaluator
        # Node data
        self.state = {
            "user_question": user_question,
            "node_content": None,
            "confidence": confidence,
        }
        self.memory = memory if memory is not None else []

        if self.node_type == NodeType.USER_QUESTION:
            self.state["node_content"] = user_question
        elif self.node_type in [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE]:
            self.state["sub_question"] = question
            self.state["sub_answer"] = answer
            self.state["reasoning"] = reasoning
            self.state["node_content"] = f"Q: {question}\nA: {answer}"
        elif self.node_type == NodeType.REPHASED_QUESTION_NODE:
            self.state["node_content"] = question
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            self.state["node_content"] = reasoning
        elif self.node_type == NodeType.FINAL_ANSWER:
            self.state["final_answer"] = answer
            self.state["detailed_answer"] = reasoning
            self.state["node_content"] = answer
        else:
            raise ValueError(f"Unknown node type: {self.node_type}")
    
    def set_rollout_id(self, rollout_id):
        self.rollout_id = rollout_id
    
    def get_node(self):
        node = copy.deepcopy(self.state)
        node['hash'] = hash(self)
        node['memory'] = self.memory if self.memory else []
        node['user_question'] = self.node_config['user_question']
        node['golden_answer'] = self.node_config['golden_answer']
        node['node_type'] = self.node_type.value
        node['depth'] = self.depth
        node['rollout_id'] = self.rollout_id if hasattr(self, 'rollout_id') else None 
        if self.parent is None:
            node['parent'] = None
        else:
            node['parent'] = hash(self.parent)
        return node

    def __hash__(self):
        node_data = self.get_node()
        node_data.pop('rollout_id', None)  # Exclude rollout_id from hash
        node_str = json.dumps(node_data, sort_keys=True)
        return int(sha256(node_str.encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if not isinstance(other, ReasoningNode):
            return False
        return hash(self) == hash(other)

    def __str__(self):
        return pprint.pformat(self.get_node(), indent=2)
    
    def is_terminal(self):
        return self.node_type == NodeType.FINAL_ANSWER or self.depth >= self.max_depth

    def is_valid_leaf(self):
        return self.node_type == NodeType.FINAL_ANSWER
    
    def print_node(self):
        print(self)

    def find_children(self):
        return
    
    def reward(self):
        return
    
    async def explore(self, question: str):
        if not question or not question.strip():
            logger.warning(f"Empty question for exploration at depth {self.depth}")
            return None
        # Generate queries for retriever
        logger.info(f"Generating queries for question: {question} at depth {self.depth}")
        agent_input = {
            'generate_fn': 'generate_queries_for_retriever',
            'question_list': question,
        }
        instance_id, _ = await self.generator.create()
        response, _, _ = await self.generator.execute(instance_id, agent_input)
        response: Optional[List[decompose_and_answer.QueriesGenerationOutput]] = response.get('generated_queries', None)
        logger.info(f"Generated queries response: {response}")
        if not response:
            logger.warning(f"No exploration output from generator for question: {question}")
            return None
        queries = [question]
        for item in response:
            if item.queries:
                queries.extend(item.queries)
        queries = list(set(queries))  # Deduplicate queries
        queries = [x.strip() for x in queries if x and x.strip()]
        
        # Retrieve
        agent_input = {
            'retrieval_query_list': queries,
            'top_k': self.node_config.get('top_k', None)
        }
        instance_id, _ = await self.retriever.create()
        response, _, _ = await self.retriever.execute(instance_id, agent_input)
        retrieval_docs = response.get('retrieval_docs', None)
        if isinstance(retrieval_docs, list) and all(isinstance(x, list) for x in retrieval_docs):
            retrieval_docs = sum(retrieval_docs, [])
            retrieval_docs = list(set(retrieval_docs))  # Deduplicate documents
        logger.info(f"Retrieved {len(retrieval_docs)} documents for queries: {queries}")
        if not retrieval_docs:
            logger.warning(f"No documents retrieved for queries: {queries}")
            return None
        
        # Filter and consolidate
        meta_info = {
                'user_question': self.node_config.get("user_question", ""),
                'question_id': self.node_config.get("question_id", ""),
                'depth': self.depth,
                'question': question
            }
        tasks = []
        for doc in retrieval_docs:
            meta_info['document'] = doc
            agent_input = {
                'question': question,
                'document': doc,
                'run_kwargs': {
                    'additional_info': meta_info,
                }
            }
            instance_id, _ = await self.extractor.create()
            tasks.append(self.extractor.execute(instance_id, agent_input))
        responses = await asyncio.gather(*tasks)
        information = []
        if not responses:
            logger.warning(f"No exploration output from extractor for question: {question} at depth {self.depth}")
            return None
        for resp in responses:
            resp: Optional[List[extract.ExtractOutput]] = resp.get('extracted_info', None)
            if not resp:
                continue
            resp: extract.ExtractOutput = resp[0]
            logger.info(f"Exploration output: {resp}")
            if resp.decision == 'relevant':
                extracted_info = list(set(resp.information))
                extracted_info = [x.strip() for x in extracted_info if x and x.strip()]
                information.extend(extracted_info)
        information: List[str] = list(set(information))  # Deduplicate information
        if not information:
            logger.warning(f"No relevant information extracted for question: {question} at depth {self.depth}")
            return None
        return information
    
    async def reflect(self, sub_question: str):
        user_question = self.node_config.get("user_question", "")
        question_id = self.node_config.get("question_id", "")
        logger.info(f"Reflecting for question ID {question_id}: {user_question} at depth {self.depth}")

        if self.memory:
            memory_str = format_memory(self.memory)
            logger.info(f"SubQuestion:\n{sub_question}\nMemory:\n{memory_str}")
            if not sub_question or not sub_question.strip():
                logger.warning(f"Empty sub-question for reflection at depth {self.depth}")
                return None
            meta_info = {
                'user_question': user_question,
                'question_id': question_id,
                'depth': self.depth,
                'memory': memory_str,
                'question': sub_question
            }
            agent_input = {
                'question': sub_question,
                'document': memory_str,
                'run_kwargs': {
                    'additional_info': meta_info,
                    'n': 3
                }
            }
            instance_id, _ = await self.extractor.create()
            response, _, _ = await self.extractor.execute(instance_id, agent_input)
            response: Optional[List[extract.ExtractOutput]] = response.get('extracted_info', None)
            if not response:
                logger.warning(f"No reflection output from extractor for question ID {question_id} at depth {self.depth}")
                return None
            response: extract.ExtractOutput = response[0]
            logger.info(f"Reflection output: {response}")
            if response.decision == 'relevant':
                extracted_info = list(set(response.information))
                extracted_info = [x.strip() for x in extracted_info if x and x.strip()]
                return extracted_info
        else:
            return None
 
    async def update_memory(self, intermediate_conclusions: List[str] = None, explored_data: List[str] = None):
        if (not intermediate_conclusions) and (not explored_data):
            return self.memory
        if intermediate_conclusions:
            intermediate_conclusions = format_memory(intermediate_conclusions)
        if explored_data:
            explored_data = format_memory(explored_data)
        memory = None
        if self.memory:
            memory = format_memory(self.memory)
        context = format_reflection_context(
            current_memory=memory,
            intermediate_conclusions=intermediate_conclusions,
            explored_data=explored_data
        )
        if not context.strip():
            logger.info(f"Empty context for memory update at depth {self.depth}")
            return self.memory
        user_question = self.node_config.get("user_question", "")
        question_id = self.node_config.get("question_id", "")
        logger.info(f"Updating memory for question ID {question_id}: {user_question} at depth {self.depth}")
        logger.info(f"Context for memory update:\n{context}")
        meta_info = {
            'user_question': user_question,
            'question_id': question_id,
            'depth': self.depth,
            'reflection_context': context
        }
        agent_input = {
            'question': user_question,
            'document': context,
            'run_kwargs': {
                'additional_info': meta_info,
                'n': 3
            }
        }
        instance_id, _ = await self.extractor.create()
        response, _, _ = await self.extractor.execute(instance_id, agent_input)
        response: Optional[List[extract.ExtractOutput]] = response.get('extracted_info', None)
        if not response:
            logger.warning(f"No memory update output from extractor for question ID {question_id} at depth {self.depth}")
            return self.memory
        response: extract.ExtractOutput = response[0]
        logger.info(f"Memory update output: {response}")
        if response.decision == 'relevant':
            extracted_info = list(set(response.information))
            extracted_info = [x.strip() for x in extracted_info if x and x.strip()]
        else:
            # if not relevant, maybe the extraction process is not good, so keep the old memory
            # Due to the context always contains the old memory, the new memory should not be empty if the extractor works well
            extracted_info = self.memory 
        return extracted_info
        
    def get_path(self):
        path: List['ReasoningNode'] = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path
    
    def get_reasoning_trace(self, path: Optional[List['ReasoningNode']] = None):
        if path is None:
            path = self.get_path()
        trace: List[str] = []
        scores: List[float] = []
        for node in path:
            if node.node_type in [NodeType.SUB_QA_NODE, NodeType.SYNTHESIS_NODE]:
                trace.append(node.state['node_content'])
                scores.append(node.state['confidence'])
            elif node.node_type == NodeType.SELF_CORRECTED_NODE:
                trace[-1] = node.state['node_content']
                scores[-1] = node.state['confidence'] 
            elif node.node_type == NodeType.FINAL_ANSWER:
                trace.append(f"Final Answer: {node.state['detailed_answer']}")
                scores.append(node.state['confidence'])
        if len(trace) == 0:
            return "", []
        trace = format_reasoning_trace(trace)
        return trace, scores

    async def generate_final_answer_node(self):
        user_question = self.node_config.get("user_question", "")
        reasoning_trace, _ = self.get_reasoning_trace()
        memory_str = format_memory(self.memory) if self.memory else ""
        explored_info = await self.explore(user_question)
        explored_str = format_memory(explored_info) if explored_info else ""
        context = format_context(memory=memory_str, reasoning_trace=reasoning_trace, explored_data=explored_str)
        logger.info(f"Generating final answer for question: {user_question} at depth {self.depth}")
        logger.info(f"Context for final answer:\n{context}")
        if not user_question or not user_question.strip():
            logger.warning(f"Empty user question for final answer generation at depth {self.depth}")
            return None, explored_info

        agent_input = {
            'generator_fn': 'finalize',
            'question_list': user_question,
            'context_list': context
        }
        instance_id, _ = await self.generator.create()
        response, _, _ = await self.generator.execute(instance_id, agent_input)
        response: Optional[List[finalize.FinalizeOutput]] = response.get('output', None)
        logger.info(f"Final answer generation output: {response}")
        if not response:
            logger.warning(f"No final answer output from generator for question: {user_question} at depth {self.depth}")
            return None, explored_info
        
        nodes: List['ReasoningNode'] = []
        all_answers = []
        for item in response:
            answer = item.answer.strip() if item.answer else ""
            detailed_answer = item.detailed_answer.strip() if item.detailed_answer else ""
            reasoning = item.reasoning.strip() if item.reasoning else ""
            confidence = convert_confidence_to_score(item.confidence)
            if not reasoning or not reasoning.strip():
                reasoning = reasoning_trace
            if not answer and not detailed_answer:
                answer = detailed_answer = reasoning
            elif not answer and detailed_answer:
                answer = detailed_answer
                detailed_answer = f"{detailed_answer}\n{reasoning}"
            elif not detailed_answer and answer:
                detailed_answer = answer
                detailed_answer = f"{answer}\n{reasoning}"
            else:
                detailed_answer = f"{detailed_answer}\n{reasoning}"
            logger.info(f"Final Answer: {answer}\nDetailed Answer: {detailed_answer}\nConfidence: {confidence}")
            if detailed_answer in all_answers:
                continue
            all_answers.append(detailed_answer)
            node = ReasoningNode(
                node_type=NodeType.FINAL_ANSWER,
                parent=self,
                answer=answer,
                reasoning=detailed_answer,
                confidence=confidence,
                **self.node_config
            )
            nodes.append(node)
        return nodes, explored_info
    
    async def answer_sub_question(self, sub_question: str, reasoning_trace: str):
        if not sub_question or not sub_question.strip():
            logger.warning(f"Empty rephrased question for subQA generation at depth {self.depth}")
            return None, None
        reflection = await self.reflect(sub_question)
        exloration = await self.explore(sub_question)
        reflection_str = format_memory(reflection) if reflection else ""
        exploration_str = format_memory(exloration) if exloration else ""
        important_info = format_context(memory=reflection_str, reasoning_trace=reasoning_trace, explored_data=exploration_str)
        logger.info(f"Generating subQA for rephrased question: {sub_question} at depth {self.depth}")
        logger.info(f"Important info for subQA:\n{important_info}")
        instance_id, _ = await self.generator.create()
        agent_input = {
            'generator_fn': 'generate_answer',
            'question_list': sub_question,
            'context_list': important_info
        }
        response, _, _ = await self.generator.execute(instance_id, agent_input)
        response: Optional[List[decompose_and_answer.AnswerOutput]] = response.get('output', None)
        logger.info(f"SubQA generation output: {response}")
        if not response:
            logger.warning(f"No subQA output from generator for rephrased question: {sub_question} at depth {self.depth}")
            return None, exploration_str
        nodes: List['ReasoningNode'] = []
        all_answers = []
        for item in response:
            sub_answer = item.answer.strip() if item.answer else ""
            reasoning = item.reasoning.strip() if item.reasoning else ""
            confidence = convert_confidence_to_score(item.confidence)
            if not sub_answer and not reasoning:
                continue
            elif not sub_answer and reasoning:
                sub_answer = reasoning
            elif not reasoning and sub_answer:
                reasoning = sub_answer
                sub_answer = f"{sub_answer}\n{reasoning}"
            logger.info(f"Sub Answer: {sub_answer}\nReasoning: {reasoning}\nConfidence: {confidence}")
            if sub_answer in all_answers:
                continue
            all_answers.append(sub_answer)
            node = ReasoningNode(
                node_type=NodeType.SUB_QA_NODE,
                parent=self,
                question=sub_question,
                answer=sub_answer,
                reasoning=reasoning,
                confidence=confidence,
                **self.node_config
            )
            nodes.append(node)
        return nodes, exloration

    async def generate_subQA_node(self):
        user_question = self.node_config.get("user_question", "")
        reasoning_trace, _ = self.get_reasoning_trace()
        if self.node_type == NodeType.REPHASED_QUESTION_NODE:
            sub_question = self.state['node_content']
            nodes, explored_info = await self.answer_sub_question(sub_question, reasoning_trace)
            return nodes, explored_info
        else:
            memory_str = format_memory(self.memory) if self.memory else ""
            context = format_context(memory=memory_str, reasoning_trace=reasoning_trace)
            logger.info(f"Decomposing question: {user_question} at depth {self.depth}")
            logger.info(f"Context for decomposition:\n{context}")
            agent_input = {
                'generator_fn': 'generate_subquestion',
                'question_list': user_question,
                'context_list': context
            }
            instance_id, _ = await self.generator.create()
            response, _, _ = await self.generator.execute(instance_id, agent_input)
            response: Optional[List[decompose_and_answer.SubquestionOutput]] = response.get('output', None)
            if not response:
                logger.warning(f"No subquestion output from generator for question: {user_question} at depth {self.depth}")
                return None, None
            logger.info(f"Decomposition output: {response}")
            answerable_main_questions = [item.answerable_main_question for item in response]
            should_answer_main = sum(answerable_main_questions) / len(answerable_main_questions)
            if should_answer_main > 0.5:
                logger.info(f"Main question is answerable, generating final answer node instead of subQA at depth {self.depth}")
                nodes, explored_info = await self.generate_final_answer_node()
                return nodes, explored_info
            sub_questions = [x.subquestion for x in response]
            sub_question = list(set(sub_questions))  # Deduplicate sub-questions
            sub_question = [x.strip() for x in sub_question if x and x.strip()]
            if not sub_question:
                logger.warning(f"No valid subquestion generated for question: {user_question} at depth {self.depth}")
                return None, None
            tasks = []
            for sq in sub_question:
                tasks.append(self.answer_sub_question(sq, reasoning_trace))
            results = await asyncio.gather(*tasks)
            nodes: List['ReasoningNode'] = []
            explored_info: List[str] = []
            for res in results:
                if res is None:
                    continue
                ns, ei = res
                if ns:
                    nodes.extend(ns)
                if ei:
                    explored_info.extend(ei)
            explored_info = list(set(explored_info)) if explored_info else None  # Deduplicate explored info
            return nodes, explored_info
    
    async def generate_rephrase_question_node(self):
        if self.node_type == NodeType.USER_QUESTION:
            question = self.state['node_content']
        elif self.node_type == NodeType.SUB_QA_NODE:
            question = self.state['sub_question']
        else:
            logger.warning(f"Invalid node type for rephrasing question at depth {self.depth}")
            return None, None
        if not question or not question.strip():
            logger.warning(f"Empty question for rephrasing at depth {self.depth}")
            return None, None
        agent_input = {
            'generator_fn': 'rephase_question',
            'question_list': question,
        }
        instance_id, _ = await self.generator.create()
        response, _, _ = await self.generator.execute(instance_id, agent_input)
        response: Optional[List[rephase_question.RephraseQuestionOutput]] = response.get('output', None)
        logger.info(f"Rephrasing output: {response}")
        if not response:
            logger.warning(f"No rephrasing output from generator for question: {question} at depth {self.depth}")
            return None, None
        nodes: List['ReasoningNode'] = []
        all_questions = []
        for item in response:
            rephrased_question = item.rephrased_question.strip() if item.rephrased_question else ""
            if not rephrased_question or not rephrased_question.strip():
                continue
            logger.info(f"Rephrased Question: {rephrased_question}")
            if rephrased_question in all_questions:
                continue
            all_questions.append(rephrased_question)
            node = ReasoningNode(
                node_type=NodeType.REPHASED_QUESTION_NODE,
                parent=self,
                question=rephrased_question,
                **self.node_config
            )
            nodes.append(node)
        return nodes, None
    
    async def generate_self_corrected_node(self):
        sub_question = self.state.get('sub_question')
        sub_answer = self.state.get('sub_answer')
        if not sub_question or not sub_question.strip():
            logger.warning(f"Empty sub-question for self-correction at depth {self.depth}")
            return None, None
        if not sub_answer or not sub_answer.strip():
            sub_answer = "Not available"
        current_step_objective = f"Verify and improve the answer to the sub-question: {sub_question}"
        reflexion = await self.reflect(current_step_objective)
        reflexion_str = format_memory(reflexion) if reflexion else ""
        explored_info = await self.explore(current_step_objective)
        exploration_str = format_memory(explored_info) if explored_info else ""
        important_info = format_context(memory=reflexion_str, explored_data=exploration_str)
        logger.info(f"Generating self-corrected reasoning for sub-question: {sub_question} at depth {self.depth}")
        logger.info(f"Important info for self-correction:\n{important_info}")
        agent_input = {
            'generator_fn': 'self_correct',
            'question_list': sub_question,
            'context_list': important_info,
            'current_answer_list': sub_answer
        }
        instance_id, _ = await self.generator.create()
        response, _, _ = await self.generator.execute(instance_id, agent_input)
        response: Optional[List[self_correct.SelfCorrectOutput]] = response.get('output', None)
        logger.info(f"Self-correction output: {response}")
        if not response:
            logger.warning(f"No self-correction output from generator for sub-question: {sub_question} at depth {self.depth}")
            return None, explored_info
        nodes: List['ReasoningNode'] = []
        all_answers = []
        for item in response:
            corrected_answer = item.reanswer.strip() if item.reanswer else ""
            reasoning = item.reasoning.strip() if item.reasoning else ""
            confidence = convert_confidence_to_score(item.confidence)
            if not corrected_answer and not reasoning:
                continue
            elif not corrected_answer and reasoning:
                corrected_answer = reasoning
            elif not reasoning and corrected_answer:
                reasoning = corrected_answer
            else:
                corrected_answer = f"{corrected_answer}\n{reasoning}"
            logger.info(f"Corrected Answer: {corrected_answer}\nReasoning: {reasoning}\nConfidence: {confidence}")
            if corrected_answer in all_answers:
                continue
            all_answers.append(corrected_answer)
            node = ReasoningNode(
                node_type=NodeType.SELF_CORRECTED_NODE,
                parent=self,
                question=sub_question,
                answer=corrected_answer,
                reasoning=reasoning,
                confidence=confidence,
                **self.node_config
            )
            nodes.append(node)
        return nodes, explored_info
    
    async def generate_synthesis_node(self):
        user_question = self.node_config.get("user_question", "")
        if not user_question or not user_question.strip():
            logger.warning(f"Empty user question for synthesis at depth {self.depth}")
            return None, None
        reasoning_trace, _ = self.get_reasoning_trace()
        memory_str = format_memory(self.memory) if self.memory else ""
        context = format_context(memory=memory_str, reasoning_trace=reasoning_trace)
        logger.info(f"Generating synthesis reasoning for question: {user_question} at depth {self.depth}")
        logger.info(f"Context for synthesis:\n{context}")
        agent_input = {
            'generator_fn': 'generate_synthesis',
            'question_list': user_question,
            'context_list': context
        }
        instance_id, _ = await self.generator.create()
        response, _, _ = await self.generator.execute(instance_id, agent_input)
        response: Optional[List[synthesize.SynthesizeOutput]] = response.get('output', None)
        logger.info(f"Synthesis output: {response}")
        if not response:
            logger.warning(f"No synthesis output from generator for question: {user_question} at depth {self.depth}")
            return None, None
        nodes: List['ReasoningNode'] = []
        all_reasonings = []
        for item in response:
            reasoning = item.synthesis.strip() if item.synthesis else ""
            confidence = convert_confidence_to_score(item.confidence)
            if not reasoning or not reasoning.strip():
                reasoning = item.reasoning.strip() if item.reasoning else ""
            if not reasoning or not reasoning.strip():
                continue
            logger.info(f"Synthesis Reasoning: {reasoning}\nConfidence: {confidence}")
            if reasoning in all_reasonings:
                continue
            all_reasonings.append(reasoning)
            if item.answerable_main_question:
                node = ReasoningNode(
                    node_type=NodeType.FINAL_ANSWER,
                    parent=self,
                    answer=reasoning,
                    reasoning=reasoning,
                    confidence=confidence,
                    **self.node_config
                )
            else:
                node = ReasoningNode(
                    node_type=NodeType.SYNTHESIS_NODE,
                    parent=self,
                    reasoning=reasoning,
                    confidence=confidence,
                    **self.node_config
                )
            nodes.append(node)
        return nodes, None
    
