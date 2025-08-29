import random
import copy
import json
from enum import Enum, unique
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple, Union
import tqdm
import pprint
from anytree import NodeMixin

from state_aware_rag.planners.MCTS.backbone import MCTS, Node
from state_aware_rag.agents.roles.generator import Generator
from state_aware_rag.agents.roles.evaluator import Evaluator
from state_aware_rag.agents.roles.extractor import Extractor
from state_aware_rag.agents.retriever_agents import RetrieverAgent

@unique
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



class ReasoningNode(Node, NodeMixin):
    """
    A node in the MCTS tree that represents a reasoning step.
    """
    def __init__(
            self,
            # Node parameters
            parent: "ReasoningNode",
            node_type: NodeType,
            depth: int,
            # Components
            generator: Generator,
            evaluator: Evaluator,
            extractor: Extractor,
            retriever: RetrieverAgent,
            # Node content
            question: Optional[str] = None,
            answer: Optional[str] = None,
            reasoning: Optional[str] = None,
            confidence: Optional[float] = None,
            memory: Optional[List[str]] = None,
            # Optional parameters
            max_depth: int = 15,
            golden_answer: Optional[Union[str, List[str]]] = None,
            user_question: Optional[str] = None,
            question_id: Optional[str] = None,
            top_k: int = 5,
            verbose: bool = False,  
            **kwargs
    ):  
        super().__init__()
        self.node_config = {
            "verbose": verbose,  # Whether to print verbose output
            "max_depth": max_depth,  # Maximum depth of the reasoning tree
            "golden_answer": golden_answer,  # The golden answer for the user question, if available
            "user_question": user_question,  # The main user question for USER_QUESTION nodes
            "question_id": question_id,  # The ID of the question, if available
            "generator": generator,  # The generator component for the node
            "evaluator": evaluator,  # The evaluator component for the node
            "extractor": extractor,  # The extractor component for the node
            "retriever": retriever,  # The retriever component for the node
            "top_k": top_k,  # The number of top-k results to retrieve from the retriever
        }
        self.verbose = verbose  # Whether to print verbose output
        self.parent = parent # Parent node in the MCTS tree, if none, this is the root node
        self.children: List["ReasoningNode"] = [] # Children nodes in the MCTS tree
        self.tree_depth = depth
        self.node_type = node_type
        # Node's agent components
        self.generator = generator
        self.retriever = retriever
        self.evaluator = evaluator
        self.extractor = extractor
        self.retriever = retriever
        self.top_k = top_k  if top_k is not None else 5
        # Node content
        self.state = {
            "user_question": None,  # The main user question for USER_QUESTION nodes
            "node_content": None,  
            "confidence": None,  # The confidence of the node content
        }
        self.memory = memory
        # Initialize the node content based on the node type
        if node_type == NodeType.USER_QUESTION:
            self.state['user_question'] = user_question
            self.state['node_content'] = user_question  # The content of the node is the user question
        elif node_type == NodeType.FINAL_ANSWER:
            self.state['node_content'] = answer
            self.state['detailed_answer'] = reasoning if reasoning is not None else ""  # Optional reasoning for FINAL_ANSWER nodes
            self.state['confidence'] = confidence if confidence is not None else 1.0  # Default confidence is 1.0 if not provided
        elif node_type == NodeType.SUB_QA_NODE:
            self.state['sub_question'] = question  # Store the sub-question
            self.state['sub_answer'] = answer  # Store the sub-answer
            self.state['node_content'] = f"{question}\n{answer}"  # Combine question and answer for SUBQUESTION nodes
            self.state['confidence'] = confidence if confidence is not None else 1.0
        elif node_type == NodeType.REPHASED_QUESTION_NODE:
            self.state['node_content'] = question
            self.state['confidence'] = 1.0 # Default confidence is 1.0 for REPHASE_QUESTION nodes
        elif node_type == NodeType.SELF_CORRECTED_NODE:
            self.state['node_content'] = f"{question}\n{answer}"
            self.state['confidence'] = confidence if confidence is not None else 1.0
        elif node_type == NodeType.SYNTHESIS_NODE:
            self.state['node_content'] = reasoning
            self.state['confidence'] = confidence if confidence is not None else 1.0
        else:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {list(NodeType)}.")
    
    def set_rollout_id(self, rollout_id: Optional[int] = None):
        """
        Set the rollout ID for the node.
        Args:
            rollout_id (str): The rollout ID to set for the node.
        """
        self.rollout_id = rollout_id
    
    def set_memory(self, memory: List[str]):
        """
        Set the memory for the node.
        Args:
            memory (List[str]): A list of strings representing the memory.
        """
        if isinstance(memory, str):
            memory = [memory]
        if isinstance(memory, list):
            memory = [item for item in memory if item]  # Remove empty strings
            memory = list(set(memory))  # Remove duplicates
            memory.sort()
        self.memory = memory
    
    def get_path(self) -> List["ReasoningNode"]:
        """
        Get the path from the root node to the current node.
        Returns:
            List[ReasoningNode]: A list of nodes from the root to the current node.
        """
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        return path[::-1] # Reverse the path to get it from root to current node
    
    def get_reasoning_trace(self, path: List["ReasoningNode"]) -> str:
        """
        Get the reasoning trace from the root node to the current node.
        Args:
            path (List[ReasoningNode]): The path from the root node to the current node.
        Returns:
            str: A string representation of the reasoning trace.
            List[float]: A list of confidence scores for each step in the reasoning trace.
        """
        reasoning_trace = []
        reasoning_scores = []
        for i, node in enumerate(path):
            if node.node_type in [NodeType.SUB_QA_NODE, NodeType.SYNTHESIS_NODE]:
                reasoning_trace.append(node.state['node_content'])
                reasoning_scores.append(node.state['confidence'])
            elif node.node_type == NodeType.SELF_CORRECTED_NODE:
                # Replace the last step in the reasoning trace with the self-corrected answer
                step_content = node.state['node_content']
                step_score = node.state['confidence']
                reasoning_trace[-1] = step_content 
                reasoning_scores[-1] = step_score
        if len(reasoning_trace) == 0:
            return None, []
        trace = ""
        for i, step in enumerate(reasoning_trace):
            trace += f"Step {i+1}: {step}\n"
        if path[-1].node_type == NodeType.FINAL_ANSWER:
            trace += f"Final answer: {path[-1].state['detailed_answer']}"
            reasoning_scores.append(path[-1].state['confidence'])
        return trace, reasoning_scores
    
    def reflect(self, sub_question: Optional[str] = None) -> List[str]:
        user_question = copy.deepcopy(self.node_config['user_question'])
        question_id = copy.deepcopy(self.node_config['question_id'])
        additional_info = {
            'user_question': user_question,
            'question_id': question_id,
            'depth': self.tree_depth,
        }
        if self.verbose:
            print(f"Reflecting on memory for sub-question: {sub_question}")
        memory_information = None
        if self.memory:
            # TODO: Try retrieving memory 
            if isinstance(self.memory, list):
                self.memory = [item for item in self.memory if item]  # Remove empty strings
                self.memory = list(set(self.memory))
                self.memory.sort()  # Sort to ensure consistent order
            memory = copy.deepcopy(self.memory)
            memory = [f"-{item}" if not item.startswith("-") else item
                      for i, item in enumerate(memory) if item.strip()] if isinstance(memory, list) else memory
            memory_knowledge = "\n".join(memory) if isinstance(memory, list) else memory
            additional_info['memory_knowledge'] = copy.deepcopy(memory)
            additional_info['question'] = sub_question
            extracted_memory = self.extractor.extract(question=sub_question, document=memory_knowledge, additional_info=additional_info, n=3)[0]
            if extracted_memory['decision'] == 'relevant':
                memory_information = extracted_memory['extracted_information']
                memory_information = [item for item in memory_information if item]  # Remove empty strings
                memory_information = list(set(memory_information))  # Remove duplicates
        if self.verbose:
            print(f"Reflected memory information: {memory_information}")
        return memory_information
    
    def explore(self, question: Optional[str] = None) -> List[str]:
        if self.verbose:
            print(f"Exploring external knowledge base for sub-question: {question}")
        additional_info = {
            'user_question': copy.deepcopy(self.node_config['user_question']),
            'question_id': copy.deepcopy(self.node_config['question_id']),
            'depth': copy.deepcopy(self.tree_depth),
            'question': question,  # The sub-question to explore
        }
        response = self.generator.generate_queries_for_retriever(question=question)
        queries_for_retriever = []
        for item in response:
            x = item.get('queries', None)
            if x is not None:
                queries_for_retriever.extend(x)
        queries_for_retriever = [q.strip() for q in queries_for_retriever if q.strip()]  # Remove empty queries
        queries_for_retriever.append(question)  # Add the sub question to the queries for retriever to prevent the case where the generated query is wrong or empty
        queries_for_retriever = list(set(queries_for_retriever))  # Remove duplicates
        retrieved_docs = self.retriever.search(query=queries_for_retriever, top_k=self.top_k)['retrieved_docs']
        if isinstance(retrieved_docs, list) and isinstance(retrieved_docs[0], list):
            retrieved_docs = sum(retrieved_docs, [])  # Flatten the list of lists
        retrieved_docs = [item['contents'] for item in retrieved_docs if 'contents' in item]  # Extract the content from the retrieved documents
        retrieved_docs = list(set(retrieved_docs))  # Remove duplicates
        retrieved_docs = [doc.strip() for doc in retrieved_docs if doc.strip()]  # Remove empty documents
        retrieved_docs.sort()  # Sort to ensure consistent order
        
        all_additional_info = []
        for doc in retrieved_docs:
            additional_info_copy = copy.deepcopy(additional_info)
            additional_info_copy['retrieved_doc'] = doc
            all_additional_info.append(additional_info_copy)
        responses = self.extractor.extract(question=[question] * len(retrieved_docs), document=retrieved_docs, additional_info=all_additional_info)
        external_information = []
        for r in responses:
            if r['decision'] == 'relevant':
                external_information.extend(r['extracted_information'])
        if self.verbose:
            print(f"Explored external information: {external_information}")
        external_information = [item for item in external_information if item]
        external_information = list(set(external_information))  # Remove duplicates
        return external_information
    
    def update_memory(
            self, 
            intermediate_conclusions: List[str], 
            step_explored_information: List[str],
            ):
        if self.verbose:
            print("Updating memory with new information.")
            print(f"Intermediate conclusions: {intermediate_conclusions}")
            print(f"Step explored information: {step_explored_information}")
        if not intermediate_conclusions and not step_explored_information:
            return self.memory  # If there is no new information, return the current memory
        
        additional_info = {
            'user_question': copy.deepcopy(self.node_config['user_question']),
            'question_id': copy.deepcopy(self.node_config['question_id']),
            'depth': copy.deepcopy(self.tree_depth),
        }
        
        user_question = self.node_config['user_question']
        raw_memory = ""
        if self.memory:
            current_memory = copy.deepcopy(self.memory)
            current_memory = [f"-{item}" if not item.startswith("-") else item
                              for i, item in enumerate(current_memory) if item.strip()] if current_memory else None
            current_memory = "\n".join(current_memory) if current_memory else None
            raw_memory += f"Current memory:\n{current_memory}\n----------\n"
        if intermediate_conclusions:
            intermediate_conclusions = [f"-{item}" if not item.startswith("-") else item
                                        for i, item in enumerate(intermediate_conclusions) if item.strip()] if intermediate_conclusions else None
            intermediate_conclusions = "\n".join(intermediate_conclusions) if intermediate_conclusions else None
            raw_memory += f"Intermediate conclusions:\n{intermediate_conclusions}\n----------\n"
        if step_explored_information:
            step_explored_information = [f"-{item}" if not item.startswith("-") else item
                                         for i, item in enumerate(step_explored_information) if item.strip()] if step_explored_information else None
            step_explored_information = "\n".join(step_explored_information) if step_explored_information else None
            raw_memory += f"Information from external KB:\n{step_explored_information}\n----------\n"
        if raw_memory == "":
            return None
        additional_info['raw_memory'] = raw_memory
        new_memory = self.extractor.extract(question=user_question, document=raw_memory, additional_info=additional_info, n=3)[0]
        new_memory = new_memory['extracted_information']
        new_memory = [item for item in new_memory if item]  # Remove empty strings
        new_memory = list(set(new_memory))  # Remove duplicates
        if self.verbose:
            print(f"New memory after update: {new_memory}")
        return new_memory

    def generate_final_answer_node(self) -> Tuple[List["ReasoningNode"], Optional[List[str]]]:
        """
        Generate a direct answer node from the current node.
        Returns:
            children (List[ReasoningNode]): A list of generated direct answer nodes.
            external_information (Optional[List[str]]): A list of important information from the external knowledge base.
        """
        if self.verbose:
            print("Generating final answer node.")
        user_question = self.node_config['user_question']
        path = self.get_path()
        reasoning_trace, _ = self.get_reasoning_trace(path)
        memory_information = copy.deepcopy(self.memory) if self.memory else None
        external_information = self.explore(question=user_question)  # Explore the external knowledge base
        important_information = ""
        if external_information:
            external_information = [f"-{item}" if not item.startswith("-") else item  # Ensure each item starts with a hyphen
                                     for item in external_information if item.strip()]  # Remove empty strings
            external_data = "\n".join(external_information)
            important_information += f"\t**Information from external KB**\n{external_data}\n----------\n"
        if memory_information:
            memory_information = [f"-{item}" if not item.startswith("-") else item
                                  for item in memory_information if item.strip()]
            memory_data = "\n".join(memory_information)
            important_information += f"\t**Memory knowledge**\n{memory_data}\n----------\n"
        if reasoning_trace:
            important_information += f"\t**Reasoning trace**\n{reasoning_trace}\n----------\n"
        if self.verbose:
            print(f"Important information for final answer generation:")
            print(important_information)
        response = self.generator.finalize(question=user_question, context=important_information)
        nodes = []
        all_answers = []
        for item in response:
            answer = item['answer'] if item['answer'] is not None and item['answer'].strip() != "" else None
            detailed_answer = item['detailed_answer'] if item['detailed_answer'] is not None and item['detailed_answer'].strip() != "" else None
            reasoning = item['reasoning'] if item['reasoning'] is not None and item['reasoning'].strip() != "" else None
            if answer is None and detailed_answer is None:
                continue
            if answer is None:
                answer = detailed_answer
            elif detailed_answer is None:
                detailed_answer = answer
            if reasoning is None:
                reasoning = reasoning_trace
            detailed_answer += f"{detailed_answer}\nReasoning: {reasoning}"
            if answer in all_answers:
                continue  # Skip duplicate answers
            if self.verbose:
                print(f"Generated final answer: {answer}\nDetailed answer: {detailed_answer}")
            all_answers.append(answer)
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.FINAL_ANSWER,
                depth=self.tree_depth + 1,
                answer=answer,
                reasoning=detailed_answer,
                confidence=item['confidence'],
                **self.node_config
            )
            nodes.append(node)
        return nodes, external_information
    
    def generate_subQA_node(self) -> Tuple[List["ReasoningNode"], Optional[List[str]]]:
        """ Generate a sub-question and answer node from the current node.
        Returns:
            children (List[ReasoningNode]): A list of generated sub-question and answer nodes.
            external_information (Optional[List[str]]): A list of important information from the external knowledge base.
        """
        if self.verbose:
            print("Generating sub-question and answer node.")
        user_question = self.node_config['user_question']
        path = self.get_path()
        reasoning_trace, _ = self.get_reasoning_trace(path)
        memory_knowledge = copy.deepcopy(self.memory) if self.memory else None
        memory_knowledge = [f"-{item}" if not item.startswith("-") else item
                            for item in memory_knowledge if item.strip()] if isinstance(memory_knowledge, list) else memory_knowledge
        memory_knowledge = "\n".join(memory_knowledge) if isinstance(memory_knowledge, list) else memory_knowledge

        # If the node is a rephrased question and its question is not the same as the user question,
        # Answer the rephrased sub-question
        if self.node_type == NodeType.REPHASED_QUESTION_NODE and self.parent.node_type != NodeType.USER_QUESTION:
            sub_question = self.state['node_content']
            memory_information = self.reflect(sub_question=sub_question)  # Reflect on the memory
            external_information = self.explore(question=sub_question)  #
            important_information = ""
            if external_information:
                external_information = [f"-{item}" if not item.startswith("-") else item
                                        for item in external_information if item.strip()]  # Remove empty strings
                external_data = "\n".join(external_information)
                important_information += f"\t**Information from external KB**\n{external_data}\n----------\n"
            if memory_information:
                memory_information = [f"-{item}" if not item.startswith("-") else item
                                      for item in memory_information if item.strip()]  # Remove empty strings
                memory_data = "\n".join(memory_information)
                important_information += f"\t**Memory knowledge**\n{memory_data}\n----------\n"
            if reasoning_trace:
                important_information += f"\t**Reasoning trace**\n{reasoning_trace}"
            if self.verbose:
                print(f"Important information for answering sub-question {sub_question}:")
                print(important_information)
            response = self.generator.generate_answer(question=sub_question, context=important_information)
            nodes = []
            all_answers = []
            for item in response:
                answer = f"{item['detailed_answer']}\nReasoning: {item['reasoning']}"
                if answer in all_answers:
                    continue  # Skip duplicate answers
                all_answers.append(answer)
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.SUB_QA_NODE,
                    depth=self.tree_depth + 1,
                    question=sub_question,
                    answer=answer,
                    confidence=item['confidence'],
                    **self.node_config
                    )
                nodes.append(node)
            return nodes, external_information
        # If the node is a sub-question, synthesized reasoning, self-corrected node, user question, or paraphrased user question,
        # Generate sub-question and answer this sub-question
        else:
            memory_data = ""
            if memory_knowledge:
                memory_data = f"\t**Memory knowledge**\n{memory_knowledge}\n----------\n"
            if reasoning_trace:
                memory_data += f"\t**Reasoning trace**\n{reasoning_trace}\n----------\n"
            if self.verbose:
                print(f"Important information for generating sub-question: {memory_data}")
            subquestion_respones = self.generator.generate_subquestion(question=user_question, context=memory_data)
            answerable_main_question = [item['answerable_main_question'] for item in subquestion_respones if item['answerable_main_question'] is not None]
            # Majority voting for answerable main question
            answerable_sub_questions = sum(answerable_main_question) / len(answerable_main_question) if len(answerable_main_question) > 0 else 0.0
            if answerable_sub_questions > 0.5:
                return self.generate_final_answer_node()  # If the answerable sub-question is more than 50%, generate direct answer node
            else:
                nodes = []
                all_external_information = []
                all_sub_questions = []
                for item in subquestion_respones:
                    sub_question = item['subquestion']
                    if sub_question is None:
                        continue
                    if sub_question.strip() == "":
                        continue
                    if sub_question in all_sub_questions:
                        continue
                    all_sub_questions.append(sub_question)
                    memory_information = self.reflect(sub_question=sub_question)  # Reflect on the memory
                    external_information = self.explore(question=sub_question)  # Explore the external knowledge base
                    important_information = ""
                    if memory_information:
                        memory_information = [f"-{item}" if not item.startswith("-") else item
                                              for item in memory_information if item.strip()]  # Remove empty strings
                        memory_data = "\n".join(memory_information)
                        important_information += f"\t**Memory knowledge**\n{memory_data}\n----------\n"
                    if external_information:
                        external_information = [f"-{item}" if not item.startswith("-") else item
                                                for item in external_information if item.strip()]
                        all_external_information.extend(external_information)
                        external_data = "\n".join(external_information)
                        important_information += f"\t**Information from external KB**\n{external_data}\n----------\n"
                    if reasoning_trace:
                        important_information += f"\t**Reasoning trace**\n{reasoning_trace}"
                    if self.verbose:
                        print(f"Important information for answering sub-question {sub_question}:")
                        print(important_information)
                    response = self.generator.generate_answer(question=sub_question, context=important_information)
                    # Get the highest confidence answer
                    answer = max(response, key=lambda x: x['confidence'])
                    if self.verbose:
                        print(f"Generated answer for sub-question {sub_question}: {answer['detailed_answer']}\nReasoning: {answer['reasoning']}")
                    node = ReasoningNode(
                        parent=self,
                        node_type=NodeType.SUB_QA_NODE,
                        depth=self.tree_depth + 1,
                        question=sub_question,
                        answer=f"{answer['detailed_answer']}\nReasoning: {answer['reasoning']}",
                        confidence=answer['confidence'],
                        **self.node_config
                    )
                    nodes.append(node)
                return nodes, all_external_information
    
    def generate_rephrase_question_node(self) -> List["ReasoningNode"]:
        if self.verbose:
            print("Generating rephrased question node.")
        assert self.node_type in [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE], "REPHASE_QUESTION nodes can only be generated from USER_QUESTION or SUB_QA_NODE nodes."
        if self.node_type == NodeType.USER_QUESTION:
            question = self.state['node_content']
        else:
            question = self.state['sub_question']
        if self.verbose:
            print(f"Rephrasing question: {question}")
        responses = self.generator.rephase_question(question=question) # Generate only one rephrased question
        children = []
        all_rephrased_questions = []
        for response in responses:
            if self.verbose:
                print(f"Rephrased question: {response['rephrased_question']}")
            if response['rephrased_question'] in all_rephrased_questions:
                continue
            all_rephrased_questions.append(response['rephrased_question'])
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.REPHASED_QUESTION_NODE,
                depth=self.tree_depth + 1,
                question=response['rephrased_question'],
                confidence=1.0,  # Default confidence is 1.0 for REPHASE_QUESTION nodes
                **self.node_config
            )
            children.append(node)
        return children
            
    def generate_self_corrected_node(self) -> Tuple[List["ReasoningNode"], Optional[List[str]]]:
        """ Generate a self-corrected node from the current node.
        Returns:
            children (List[ReasoningNode]): A list of generated self-corrected nodes.
            external_information (Optional[List[str]]): A list of important information from the external knowledge base.
        """
        if self.verbose:
            print("Generating self-corrected node.")
        assert self.node_type == NodeType.SUB_QA_NODE, "SELF_CORRECTED_NODE can only be generated from SUB_QA_NODE nodes."
        sub_question = self.state['sub_question']
        sub_answer = self.state['sub_answer']
        current_step_objective = f"Verify the answer for the question: {sub_question}\nAnswer: {sub_answer}"
        user_question = self.node_config['user_question']

        memory_information = self.reflect(sub_question=current_step_objective)  # Reflect on the memory
        external_information = self.explore(question=current_step_objective)  # Explore the external knowledge base
        important_information = ""
        if memory_information:
            memory_information = [f"-{item}" if not item.startswith("-") else item
                                  for item in memory_information if item.strip()]  # Remove empty strings
            memory_data = "\n".join(memory_information)
            important_information += f"\t**Memory knowledge**\n{memory_data}\n----------\n"
        if external_information:
            external_information = [f"-{item}" if not item.startswith("-") else item
                                    for item in external_information if item.strip()]  # Remove empty strings
            external_data = "\n".join(external_information)
            important_information += f"\t**Information from external KB**\n{external_data}\n----------\n"
        if self.verbose:
            print(f"Important information for self-correcting the answer for sub-question {sub_question}, sub-answer {sub_answer}:")
            print(important_information)
        response = self.generator.self_correct(question=sub_question, current_answer=sub_answer, context=important_information)
        nodes = []
        all_reanswer = []
        for item in response:
            if self.verbose:
                print(f"Generated self-corrected answer: {item['reanswer']}\nReasoning: {item['reasoning']}")
            reanswer=f"{item['reanswer']}\nReasoning: {item['reasoning']}"
            if reanswer in all_reanswer:
                continue
            all_reanswer.append(reanswer)
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.SELF_CORRECTED_NODE,
                depth=self.tree_depth + 1,
                question=sub_question,
                answer=reanswer,
                confidence=item['confidence'],
                **self.node_config
            )
            nodes.append(node)
        return nodes, external_information
    
    def generate_synthesis_node(self) -> List["ReasoningNode"]:
        """ Generate a synthesis node from the current node.
        Returns:
            children (List[ReasoningNode]): A list of generated synthesis nodes.
        """
        if self.verbose:
            print("Generating synthesis node.")
        assert self.node_type in [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE], "SYNTHESIS_NODE can only be generated from SUB_QA_NODE or SELF_CORRECTED_NODE nodes."
        user_question = self.node_config['user_question']
        path = self.get_path()
        reasoning_trace, _ = self.get_reasoning_trace(path)
        memory_information = self.reflect(sub_question=user_question)  # Reflect on the memory
        important_information = ""
        if memory_information:
            memory_information = [f"-{item}" if not item.startswith("-") else item
                                  for item in memory_information if item.strip()]  # Remove empty strings
            memory_data = "\n".join(memory_information)
            important_information += f"\t**Memory knowledge**\n{memory_data}\n----------\n"
        if reasoning_trace:
            important_information += f"\t**Reasoning trace**\n{reasoning_trace}\n----------\n"
        if self.verbose:
            print(f"Important information for generating synthesis node:")
            print(important_information)
        response = self.generator.generate_synthesis(question=user_question, context=important_information)
        nodes = []
        all_syntheses = []
        for item in response:
            if not item['synthesis'] or item['synthesis'].strip() == "":
                continue
            if item['synthesis'] in all_syntheses:
                continue
            all_syntheses.append(item['synthesis'])
            if self.verbose:
                print(f"Generated synthesis: {item['synthesis']}")
            answerable_main_question = item['answerable_main_question']
            if answerable_main_question:
                answer = item['synthesis']
                detailed_answer = item['synthesis']
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.FINAL_ANSWER,
                    depth=self.tree_depth + 1,
                    answer=answer,
                    reasoning=detailed_answer,
                    confidence=item['confidence'],
                    **self.node_config
                )
            else:
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.SYNTHESIS_NODE,
                    depth=self.tree_depth + 1,
                    reasoning=item['synthesis'],
                    confidence=item['confidence'],
                    **self.node_config
                )
            nodes.append(node)
        return nodes

    def generate_children(self):
        """
        Find and generate children nodes based on the current node type.
        Returns:
            List[ReasoningNode]: A list of generated child nodes based on the current node type.
        """
        explored_information = []
        intermediate_conclusions = []
        if self.tree_depth == self.node_config['max_depth'] - 1:
            # If the maximum depth is reached, generate a final answer node
            final_answer_nodes, external_information = self.generate_final_answer_node()
            explored_information += external_information
            children = final_answer_nodes
        elif self.node_type == NodeType.USER_QUESTION:
            final_answer_nodes, external_information = self.generate_final_answer_node()
            explored_information += external_information
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            rephrase_nodes = self.generate_rephrase_question_node()
            children = sub_qa_nodes + rephrase_nodes + final_answer_nodes
        elif self.node_type == NodeType.FINAL_ANSWER:
            # If the node is a final answer node, it has no children
            raise ValueError("Final answer nodes cannot have children.")
        elif self.node_type == NodeType.SUB_QA_NODE:
            rephrase_nodes = self.generate_rephrase_question_node()
            self_corrected_nodes, external_information = self.generate_self_corrected_node()
            explored_information += external_information
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            synthesis_nodes = self.generate_synthesis_node()
            rephrase_nodes = self.generate_rephrase_question_node()
            children = rephrase_nodes + self_corrected_nodes + synthesis_nodes + sub_qa_nodes
        elif self.node_type == NodeType.REPHASED_QUESTION_NODE:
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            children = sub_qa_nodes
        elif self.node_type == NodeType.SELF_CORRECTED_NODE:
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            synthesis_nodes = self.generate_synthesis_node()
            children = sub_qa_nodes + synthesis_nodes
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            children = sub_qa_nodes
        else:
            raise ValueError(f"Invalid node type: {self.node_type}. Must be one of {list(NodeType)}.")
        children = list(set(children))  # Remove duplicates
        children = [child for child in children if child]
        
        if self.verbose:
            print(f"Memory at depth: {self.tree_depth}:")
            print(self.memory)
        assert isinstance(explored_information, list), "Explored information must be a list."
        assert isinstance(intermediate_conclusions, list), "Intermediate conclusions must be a list."
        explored_information = [item for item in explored_information if item]  # Remove empty strings
        explored_information = list(set(explored_information))  # Remove duplicates
        # Sort to ensure consistent order
        explored_information.sort()
        intermediate_conclusions = [item for item in intermediate_conclusions if item]  # Remove empty strings
        intermediate_conclusions = list(set(intermediate_conclusions))  # Remove duplicates
        intermediate_conclusions.sort()
        new_memory = self.update_memory(intermediate_conclusions=intermediate_conclusions, step_explored_information=explored_information)
        # assert len(children) > 0, f"No children generated for node: {self.print_node()}"
        return children, new_memory
    
    def find_children(self, rollout_id: Optional[int] = None):
        if self.children:
            return self.children
        _, new_memory = self.generate_children()
        # self.children = children
        for child in self.children:
            child.set_memory(new_memory)
            child.set_rollout_id(rollout_id)
        return self.children

    def is_valid_leaf(self) -> bool:
        """
        Check if the current node is a valid leaf node.
        Returns:
            bool: True if the node is a valid leaf, False otherwise.
        """
        if self.node_type == NodeType.FINAL_ANSWER:
            return True
        return False
    
    def is_terminal(self) -> bool:
        return self.tree_depth > self.node_config['max_depth'] or self.is_valid_leaf()
    
    def reward(self) -> float:
        """
        Calculate the reward for the current node.
        The reward is based on the node type and content.
        Returns:
            float: The reward value for the node.
        """
        assert self.is_valid_leaf(), "Reward can only be calculated for valid leaf nodes."
        user_question = self.node_config['user_question']
        golden_answer = self.node_config['golden_answer']
        answer = self.state['node_content'] 
        detailed_answer = self.state['detailed_answer'] if self.state.get('detailed_answer') else answer
        answer_confidence = self.state['confidence']
        path = self.get_path()
        _, reasoning_scores = self.get_reasoning_trace(path)

        # Answer reward
        answer_side_reward = None
        if golden_answer:
            em_score = self.evaluator.evaluate_with_em(golden_answer, answer)[0]
            judge_score = self.evaluator.judge_answer(user_question=user_question, system_answer=detailed_answer, correct_answer=golden_answer)[0]
            if judge_score is None:
                judge_score = 0.0
            judge_score = float(judge_score) / 10
            answer_side_reward = 0.5*(em_score + judge_score) * answer_confidence
        else:
            judge_score = self.evaluator.judge_answer(user_question=user_question, system_answer=detailed_answer, correct_answer=None)[0]
            if judge_score is None:
                judge_score = 0.0
            judge_score = float(judge_score) / 10
            answer_side_reward = judge_score * answer_confidence
        
        # Reasoning confidence reward
        reasoning_side_reward = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
        reward = 0.0
        if answer_side_reward:
            reward = 0.75 * answer_side_reward + 0.25 * reasoning_side_reward
        else:
            reward = reasoning_side_reward
        return reward
        
    def __hash__(self):
        node = copy.deepcopy(self.state)
        node['memory'] = self.memory if self.memory else [] 
        node['node_type'] = self.node_type.value
        node['depth'] = self.tree_depth
        if self.parent is None:
            return  0  # If the node has no parent, return 0 as the hash value
        else:
            parent_hash = hash(self.parent) # Use the hash of the parent node to ensure uniqueness
        node['parent'] = parent_hash
        node_content_str = json.dumps(node, sort_keys=True)  # Convert the node content to a string for hashing
        return int(sha256(node_content_str.encode('utf-8')).hexdigest(), 16)
    
    def __eq__(self, other):
        """
        Equality check for the ReasoningNode.
        Two nodes are considered equal if they have same hash value, i.e., same content, type, depth, and parent.
        Args:
            other (ReasoningNode): The other node to compare with.
        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        if not isinstance(other, ReasoningNode):
            return False
        same_hash = hash(self) == hash(other)
        same_type = self.node_type == other.node_type
        same_depth = self.tree_depth == other.tree_depth
        return same_hash and same_type and same_depth
    
    def __str__(self):
        node_id = hash(self)
        node_type = self.node_type.value
        return f"{node_type}-{node_id}"

    def get_node(self):
        """
        Print the node content in a readable format.
        """
        node = copy.deepcopy(self.state)
        node['hash'] = hash(self)
        node['memory'] = self.memory if self.memory else []
        node['user_question'] = self.node_config['user_question']
        node['golden_answer'] = self.node_config['golden_answer']
        node['node_type'] = self.node_type.value
        node['depth'] = self.tree_depth
        node['rollout_id'] = self.rollout_id if hasattr(self, 'rollout_id') else None
        if self.parent is None:
            node['parent'] = None
        else:
            node['parent'] = hash(self.parent)
        path = self.get_path()
        reasoning_path, reasoning_scores = self.get_reasoning_trace(path)
        full_reasoning_path = []
        for i, n in enumerate(path):
            node_state = copy.deepcopy(n.state)
            node_state['node_type'] = n.node_type.value
            node_state['depth'] = n.tree_depth
            node_state['memory'] = n.memory if n.memory else []
            full_reasoning_path.append(node_state)
        node['full_reasoning_path'] = full_reasoning_path
        node['reasoning_path'] = reasoning_path
        node['reasoning_scores'] = reasoning_scores
        return node
    
    def print_node(self) -> str:
        """
        Print the node content in a readable format.
        Returns:
            str: A string representation of the node content.
        """
        node = self.get_node()
        node = copy.deepcopy(node)
        node.pop('full_reasoning_path')
        pprint.pprint(node, indent=4, width=120)



