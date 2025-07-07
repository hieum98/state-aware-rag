import random
import copy
import json
from enum import Enum, unique
from hashlib import sha256
from typing import List, Optional, Tuple, Union
import tqdm
import pprint

from planners.MCTS.backbone import MCTS, Node
from agents.roles.generator import Generator
from agents.roles.evaluator import Evaluator
from agents.roles.extractor import Extractor
from agents.retriever_agents import RetrieverAgent

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



class ReasoningNode(Node):
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
            **kwargs
    ):  
        super().__init__()
        self.node_config = {
            "max_depth": max_depth,  # Maximum depth of the reasoning tree
            "golden_answer": golden_answer,  # The golden answer for the user question, if available
            "user_question": user_question,  # The main user question for USER_QUESTION nodes
            "generator": generator,  # The generator component for the node
            "evaluator": evaluator,  # The evaluator component for the node
            "extractor": extractor,  # The extractor component for the node
            "retriever": retriever,  # The retriever component for the node
        }
        self.parent = parent # Parent node in the MCTS tree, if none, this is the root node
        self.children: List["ReasoningNode"] = [] # Children nodes in the MCTS tree
        self.depth = depth
        self.node_type = node_type
        # Node's agent components
        self.generator = generator
        self.retriever = retriever
        self.evaluator = evaluator
        self.extractor = extractor
        self.retriever = retriever
        # Node content
        self.state = {
            "user_question": None,  # The main user question for USER_QUESTION nodes
            # The content of the intermediate and the final nodes
            # Which can be final answer for FINAL_ANSWER nodes,
            # subquestion and subanswer for SUBQUESTION nodes and SELF_CORRECTED_NODE nodes,
            # rephased question for REPHASE_QUESTION nodes,
            # synthesized reasoning for SYNTHESIS nodes,
            "node_content": None,  
            "confidence": None,  # The confidence of the node content
        }
        self.memory = memory
        # Initialize the node content based on the node type
        if node_type == NodeType.USER_QUESTION:
            assert user_question is not None, "User question must be provided for USER_QUESTION nodes."
            self.state['user_question'] = user_question
            self.state['node_content'] = user_question  # The content of the node is the user question
        elif node_type == NodeType.FINAL_ANSWER:
            assert answer is not None, "Answer must be provided for FINAL_ANSWER nodes."
            self.state['node_content'] = answer
            self.state['confidence'] = confidence if confidence is not None else 1.0  # Default confidence is 1.0 if not provided
        elif node_type == NodeType.SUB_QA_NODE:
            assert question is not None, "Question must be provided for SUBQUESTION nodes."
            assert answer is not None, "Answer must be provided for SUBQUESTION nodes."
            self.state['sub_question'] = question  # Store the sub-question
            self.state['sub_answer'] = answer  # Store the sub-answer
            self.state['node_content'] = f"{question}\n{answer}"  # Combine question and answer for SUBQUESTION nodes
            self.state['confidence'] = confidence if confidence is not None else 1.0
        elif node_type == NodeType.REPHASED_QUESTION_NODE:
            assert self.parent.node_type in [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE], "REPHASE_QUESTION nodes can only be generated from USER_QUESTION or SUB_QA_NODE nodes."
            assert question is not None, "Question must be provided for REPHASE_QUESTION nodes."
            self.state['node_content'] = question
            self.state['confidence'] = 1.0 # Default confidence is 1.0 for REPHASE_QUESTION nodes
        elif node_type == NodeType.SELF_CORRECTED_NODE:
            assert question is not None, "Question must be provided for SELF_CORRECTED_NODE nodes."
            assert answer is not None, "Answer must be provided for SELF_CORRECTED_NODE nodes."
            assert self.parent.node_type == NodeType.SUB_QA_NODE, "SELF_CORRECTED_NODE can only be generated from SUB_QA_NODE nodes."
            self.state['node_content'] = f"{question}\n{answer}"
            self.state['confidence'] = confidence if confidence is not None else 1.0
        elif node_type == NodeType.SYNTHESIS_NODE:
            assert reasoning is not None, "Reasoning must be provided for SYNTHESIS nodes."
            assert self.parent.node_type in [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE], "SYNTHESIS_NODE can only be generated from SUB_QA_NODE or SELF_CORRECTED_NODE nodes."
            self.state['node_content'] = reasoning
            self.state['confidence'] = confidence if confidence is not None else 1.0
        else:
            raise ValueError(f"Invalid node type: {node_type}. Must be one of {list(NodeType)}.")
    
    def set_memory(self, memory: List[str]):
        """
        Set the memory for the node.
        Args:
            memory (List[str]): A list of strings representing the memory.
        """
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
            if node.node_type == NodeType.SUB_QA_NODE:
                reasoning_trace.append(node.state['node_content'])
                reasoning_scores.append(node.state['confidence'])
            elif node.node_type == NodeType.SELF_CORRECTED_NODE:
                # Replace the last step in the reasoning trace with the self-corrected answer
                step_content = node.state['node_content']
                step_score = node.state['confidence']
                reasoning_trace[-1] = step_content 
                reasoning_scores[-1] = step_score
            elif node.node_type == NodeType.SYNTHESIS_NODE:
                reasoning_trace.append(node.state['node_content'])
                reasoning_scores.append(node.state['confidence'])
        if len(reasoning_trace) == 0:
            return None, []
        trace = ""
        trace += f"Reasoning trace for question '{self.state['user_question']}':\n"
        for i, step in enumerate(reasoning_trace):
            trace += f"Step {i+1}: {step}\n"
        if path[-1].node_type == NodeType.FINAL_ANSWER:
            trace += f"Final answer: {path[-1].state['node_content']}."
            reasoning_scores.append(path[-1].state['confidence'])
        return trace, reasoning_scores
    
    def reflect(self, sub_question: Optional[str] = None) -> List[str]:
        memory_information = None
        if self.memory:
            # TODO: Try retrieving memory 
            memory_knowledge = "\n".join(self.memory)
            extracted_memory = self.extractor.extract(question=sub_question, current_step_objective=sub_question, document=memory_knowledge)[0]
            if extracted_memory['decision'] == 'relevant':
                memory_information = extracted_memory['extracted_information']
        return memory_information
    
    def explore(self, user_question: str, sub_question: Optional[str] = None) -> List[str]:
        queries_for_retriever = self.generator.generate_queries_for_retriever(question=sub_question)[0]['queries']
        retrieved_docs = self.retriever.search(query=queries_for_retriever, top_k=64, reranker_top_k=3)['retrieved_docs']
        if isinstance(retrieved_docs, list) and isinstance(retrieved_docs[0], list):
            retrieved_docs = sum(retrieved_docs, [])  # Flatten the list of lists
        # TODO: Try batch extraction
        retrieved_information = ""
        for i, doc in enumerate(retrieved_docs):
            retrieved_information += f"Retrieved information {i+1}:\n{doc}\n"
        extracted_retrieval_information = self.extractor.extract(question=user_question, current_step_objective=sub_question, document=retrieved_information)[0]
        external_information = []
        if extracted_retrieval_information['decision'] == 'relevant':
            external_information = extracted_retrieval_information['extracted_information']
        return external_information
    
    def update_memory(
            self, 
            intermediate_conclusions: List[str], 
            step_explored_information: List[str],
            ):
        user_question = self.node_config['user_question']
        current_memory = "\n".join(self.memory) if self.memory else None
        intermediate_conclusions = "\n".join(intermediate_conclusions) if intermediate_conclusions else None
        step_explored_information = "\n".join(step_explored_information) if step_explored_information else None
        raw_memory = ""
        if current_memory:
            raw_memory += f"Current memory:\n{current_memory}\n----------\n"
        if intermediate_conclusions:
            raw_memory += f"Intermediate conclusions:\n{intermediate_conclusions}\n----------\n"
        if step_explored_information:
            raw_memory += f"Retrieved information from external knowledge base:\n{step_explored_information}\n----------\n"
        assert raw_memory != "", "Memory cannot be empty."
        new_memory = self.extractor.extract(question=user_question, current_step_objective=user_question, document=raw_memory)[0]
        new_memory = new_memory['extracted_information']
        return new_memory

    def generate_final_answer_node(self) -> Tuple[List["ReasoningNode"], Optional[List[str]]]:
        """
        Generate a direct answer node from the current node.
        Returns:
            children (List[ReasoningNode]): A list of generated direct answer nodes.
            external_information (Optional[List[str]]): A list of important information from the external knowledge base.
        """
        user_question = self.node_config['user_question']
        path = self.get_path()
        reasoning_trace, _ = self.get_reasoning_trace(path)
        memory_information = self.reflect(sub_question=user_question)  # Reflect on the memory
        external_information = self.explore(user_question=user_question, sub_question=user_question)  # Explore the external knowledge base
        important_information = ""
        if memory_information:
            memory_data = "\n".join(memory_information)
            important_information += f"\t**Retrieved information from memory**\n{memory_data}\n----------\n"
        if external_information:
            external_data = "\n".join(external_information)
            important_information += f"\t**Retrieved information from external knowledge base**\n{external_data}\n----------\n"
        important_information += f"\t**Reasoning trace**\n{reasoning_trace}"
        response = self.generator.finalize(question=user_question, context=important_information)
        nodes = []
        for item in response:
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.FINAL_ANSWER,
                depth=self.depth + 1,
                answer=item['answer'],
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
        user_question = self.node_config['user_question']
        path = self.get_path()
        reasoning_trace, _ = self.get_reasoning_trace(path)
        memory_knowledge = "\n".join(self.memory) if self.memory else None

        # If the node is a rephrased question and its question is not the same as the user question,
        # Answer the rephrased sub-question
        if self.node_type == NodeType.REPHASED_QUESTION_NODE and self.parent.node_type != NodeType.USER_QUESTION:
            sub_question = self.state['node_content']
            memory_information = self.reflect(sub_question=sub_question)  # Reflect on the memory
            external_information = self.explore(user_question=user_question, sub_question=sub_question)  #
            important_information = ""
            if memory_information:
                memory_data = "\n".join(memory_information)
                important_information += f"\t**Retrieved information from memory**\n{memory_data}\n----------\n"
            if external_information:
                external_data = "\n".join(external_information)
                important_information += f"\t**Retrieved information from external knowledge base**\n{external_data}\n----------\n"
            if reasoning_trace:
                important_information += f"\t**Reasoning trace**\n{reasoning_trace}"
            response = self.generator.generate_answer(question=sub_question, context=important_information)
            nodes = []
            for item in response:
                answer = f"{item['detailed_answer']}.\nReasoning: {item['reasoning']}"
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.SUB_QA_NODE,
                    depth=self.depth + 1,
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
            subquestion_respones = self.generator.generate_subquestion(question=user_question, context=memory_data)
            answerable_main_question = [item['answerable_main_question'] for item in subquestion_respones]
            # Majority voting for answerable main question
            answerable_sub_questions = sum(answerable_main_question) / len(answerable_main_question) 
            if answerable_sub_questions > 0.5:
                return self.generate_final_answer_node()  # If the answerable sub-question is more than 50%, generate direct answer node
            else:
                nodes = []
                all_external_information = []
                for item in subquestion_respones:
                    sub_question = item['subquestion']
                    if sub_question.strip() == "":
                        continue
                    memory_information = self.reflect(sub_question=sub_question)  # Reflect on the memory
                    external_information = self.explore(user_question=user_question, sub_question=sub_question)  # Explore the external knowledge base
                    important_information = ""
                    if memory_information:
                        memory_data = "\n".join(memory_information)
                        important_information += f"\t**Retrieved information from memory**\n{memory_data}\n----------\n"
                    if external_information:
                        all_external_information.extend(external_information)
                        external_data = "\n".join(external_information)
                        important_information += f"\t**Retrieved information from external knowledge base**\n{external_data}\n----------\n"
                    if reasoning_trace:
                        important_information += f"\t**Reasoning trace**\n{reasoning_trace}"
                    response = self.generator.generate_answer(question=sub_question, context=important_information)
                    # Get the highest confidence answer
                    answer = max(response, key=lambda x: x['confidence'])
                    node = ReasoningNode(
                        parent=self,
                        node_type=NodeType.SUB_QA_NODE,
                        depth=self.depth + 1,
                        question=sub_question,
                        answer=f"{answer['detailed_answer']}.\nReasoning: {answer['reasoning']}",
                        confidence=answer['confidence'],
                        **self.node_config
                    )
                    nodes.append(node)
                return nodes, all_external_information
    
    def generate_rephrase_question_node(self) -> List["ReasoningNode"]:
        assert self.node_type in [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE], "REPHASE_QUESTION nodes can only be generated from USER_QUESTION or SUB_QA_NODE nodes."
        if self.node_type == NodeType.USER_QUESTION:
            question = self.state['node_content']
        else:
            question = self.state['sub_question']
        response = self.generator.rephase_question(question=question, n=1)[0] # Generate only one rephrased question
        node = ReasoningNode(
            parent=self,
            node_type=NodeType.REPHASED_QUESTION_NODE,
            depth=self.depth + 1,
            question=response['rephrased_question'],
            confidence=1.0,  # Default confidence is 1.0 for REPHASE_QUESTION nodes
            **self.node_config
        )
        return [node]
            
    def generate_self_corrected_node(self) -> Tuple[List["ReasoningNode"], Optional[List[str]]]:
        """ Generate a self-corrected node from the current node.
        Returns:
            children (List[ReasoningNode]): A list of generated self-corrected nodes.
            external_information (Optional[List[str]]): A list of important information from the external knowledge base.
        """
        assert self.node_type == NodeType.SUB_QA_NODE, "SELF_CORRECTED_NODE can only be generated from SUB_QA_NODE nodes."
        sub_question = self.state['sub_question']
        sub_answer = self.state['sub_answer']
        current_step_objective = f"Verify the answer for the question: {sub_question}.\nAnswer: {sub_answer}"
        user_question = self.node_config['user_question']

        memory_information = self.reflect(sub_question=current_step_objective)  # Reflect on the memory
        external_information = self.explore(user_question=user_question, sub_question=current_step_objective)  # Explore the external knowledge base
        important_information = ""
        if memory_information:
            memory_data = "\n".join(memory_information)
            important_information += f"\t**Retrieved information from memory**\n{memory_data}\n----------\n"
        if external_information:
            external_data = "\n".join(external_information)
            important_information += f"\t**Retrieved information from external knowledge base**\n{external_data}\n----------\n"
        response = self.generator.self_correct(question=sub_question, answer=sub_answer, context=important_information)
        nodes = []
        for item in response:
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.SELF_CORRECTED_NODE,
                depth=self.depth + 1,
                question=sub_question,
                answer=f"{item['reanswer']}.\nReasoning: {item['reasoning']}",
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
        assert self.node_type in [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE], "SYNTHESIS_NODE can only be generated from SUB_QA_NODE or SELF_CORRECTED_NODE nodes."
        user_question = self.node_config['user_question']
        path = self.get_path()
        reasoning_trace, _ = self.get_reasoning_trace(path)
        memory_information = self.reflect(sub_question=user_question)  # Reflect on the memory
        important_information = ""
        if memory_information:
            memory_data = "\n".join(memory_information)
            important_information += f"\t**Retrieved information from memory**\n{memory_data}\n----------\n"
        if reasoning_trace:
            important_information += f"\t**Reasoning trace**\n{reasoning_trace}\n----------\n"
        response = self.generator.generate_synthesis(question=user_question, context=important_information)
        nodes = []
        for item in response:
            answerable_main_question = item['answerable_main_question']
            if answerable_main_question:
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.FINAL_ANSWER,
                    depth=self.depth + 1,
                    answer=item['synthesis'],
                    confidence=item['confidence'],
                    **self.node_config
                )
            else:
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.SYNTHESIS_NODE,
                    depth=self.depth + 1,
                    reasoning=item['synthesis'],
                    confidence=item['confidence'],
                    **self.node_config
                )
            nodes.append(node)
        return nodes

    def find_children(self):
        """
        Find and generate children nodes based on the current node type.
        Returns:
            List[ReasoningNode]: A list of generated child nodes based on the current node type.
        """
        explored_information = []
        intermediate_conclusions = []
        if self.depth == self.node_config['max_depth'] - 1:
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
            children = final_answer_nodes + sub_qa_nodes + rephrase_nodes
        elif self.node_type == NodeType.FINAL_ANSWER:
            # If the node is a final answer node, it has no children
            raise ValueError("Final answer nodes cannot have children.")
        elif self.node_type == NodeType.SUB_QA_NODE:
            final_answer_nodes, external_information = self.generate_final_answer_node()
            explored_information += external_information
            rephrase_nodes = self.generate_rephrase_question_node()
            self_corrected_nodes, external_information = self.generate_self_corrected_node()
            explored_information += external_information
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            synthesis_nodes = self.generate_synthesis_node()
            rephrase_nodes = self.generate_rephrase_question_node()
            children = final_answer_nodes + rephrase_nodes + self_corrected_nodes + synthesis_nodes + sub_qa_nodes
        elif self.node_type == NodeType.REPHASED_QUESTION_NODE:
            final_answer_nodes, external_information = self.generate_final_answer_node()
            explored_information += external_information
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            children = final_answer_nodes + sub_qa_nodes
        elif self.node_type == NodeType.SELF_CORRECTED_NODE:
            final_answer_nodes, external_information = self.generate_final_answer_node()
            explored_information += external_information
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            synthesis_nodes = self.generate_synthesis_node()
            children = final_answer_nodes + sub_qa_nodes + synthesis_nodes
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            final_answer_nodes, external_information = self.generate_final_answer_node()
            explored_information += external_information
            sub_qa_nodes, external_information = self.generate_subQA_node()
            explored_information += external_information
            children = final_answer_nodes + sub_qa_nodes
        else:
            raise ValueError(f"Invalid node type: {self.node_type}. Must be one of {list(NodeType)}.")
        
        new_memory = self.update_memory(intermediate_conclusions=intermediate_conclusions, step_explored_information=explored_information)
        for child in children:
            child.set_memory(new_memory)
        assert len(children) > 0, f"No children generated for node type: {self.print_node()}"
        self.children = children
        return children
    
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
        return self.depth > self.node_config['max_depth'] or self.is_valid_leaf()
    
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
        answer_confidence = self.state['confidence']
        path = self.get_path()
        _, reasoning_scores = self.get_reasoning_trace(path)

        # Answer reward
        answer_side_reward = None
        if golden_answer:
            em_score = self.evaluator.evaluate_with_em(golden_answer, answer)[0]
            judge_score = self.evaluator.evaluate_final_answer(question=user_question, correct_answer=golden_answer, answer=answer)
            answer_side_reward = 0.5*(em_score + judge_score) * answer_confidence
        
        # Reasoning confidence reward
        reasoning_side_reward = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
        reward = 0.0
        if answer_side_reward:
            reward = 0.75 * answer_side_reward + 0.25 * reasoning_side_reward
        else:
            reward = reasoning_side_reward

        return reward
    
    def find_random_child(self):
        if self.is_terminal():
            return None  # If the node is terminal, return None
        node_children = self.find_children()
        random_child = random.choice(node_children) if node_children else None
        return random_child  # Return a random child node, or None if there are no children
        
    def __hash__(self):
        node = copy.deepcopy(self.state)
        node['memory'] = self.memory if self.memory else [] 
        node['node_type'] = self.node_type.value
        node['depth'] = self.depth
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
        same_depth = self.depth == other.depth
        return same_hash and same_type and same_depth
        
    def get_node(self):
        """
        Print the node content in a readable format.
        """
        node = copy.deepcopy(self.state)
        node['memory'] = self.memory if self.memory else []
        node['node_type'] = self.node_type.value
        node['depth'] = self.depth
        if self.parent is None:
            node['parent'] = None
        else:
            node['parent'] = hash(self.parent)
        path = self.get_path()
        reasoning_path, reasoning_scores = self.get_reasoning_trace(path)
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
        pprint.pprint(node, indent=4, width=120)



