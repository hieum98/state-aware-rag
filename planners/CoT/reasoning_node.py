import random
import copy
import json
from enum import Enum, unique
from hashlib import sha256
from typing import List, Optional, Tuple, Union
import tqdm
import pprint
from anytree import NodeMixin

from planners.CoT.backbone import Node
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
            # The content of the intermediate and the final nodes
            # Which can be final answer for FINAL_ANSWER nodes,
            # subquestion and subanswer for SUBQUESTION nodes
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
            if node.node_type == NodeType.SUB_QA_NODE:
                reasoning_trace.append(node.state['node_content'])
                reasoning_scores.append(node.state['confidence'])
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
            memory = [f"- {item}" for i, item in enumerate(memory)] if isinstance(memory, list) else memory
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
            current_memory = [f"- {item}" for i, item in enumerate(current_memory)] if current_memory else None
            current_memory = "\n".join(current_memory) if current_memory else None
            raw_memory += f"Current memory:\n{current_memory}\n----------\n"
        if intermediate_conclusions:
            intermediate_conclusions = [f"- {item}" for i, item in enumerate(intermediate_conclusions)] if intermediate_conclusions else None
            intermediate_conclusions = "\n".join(intermediate_conclusions) if intermediate_conclusions else None
            raw_memory += f"Intermediate conclusions:\n{intermediate_conclusions}\n----------\n"
        if step_explored_information:
            step_explored_information = [f"- {item}" for i, item in enumerate(step_explored_information)] if step_explored_information else None
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
            external_information = [f"- {item}" for item in external_information if item]  # Remove empty strings
            external_data = "\n".join(external_information)
            important_information += f"\t**Information from external KB**\n{external_data}\n----------\n"
        if memory_information:
            memory_information = [f"- {item}" for item in memory_information if item]
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
        memory_knowledge = [f"- {item}" for item in memory_knowledge if item] if isinstance(memory_knowledge, list) else memory_knowledge
        memory_knowledge = "\n".join(memory_knowledge) if isinstance(memory_knowledge, list) else memory_knowledge

        memory_data = ""
        if memory_knowledge:
            memory_data = f"\t**Memory knowledge**\n{memory_knowledge}\n----------\n"
        if reasoning_trace:
            memory_data += f"\t**Reasoning trace**\n{reasoning_trace}\n----------\n"
        if self.verbose:
            print(f"Important information for generating sub-question: {memory_data}")
        # For CoT, only generate one sub-question per step
        subquestion_respones = self.generator.generate_subquestion(question=user_question, context=memory_data, n=1)
        answerable_main_question = [item['answerable_main_question'] for item in subquestion_respones if item['answerable_main_question'] is not None]
        # Majority voting for answerable main question
        answerable_main_question = sum(answerable_main_question) / len(answerable_main_question) if len(answerable_main_question) > 0 else 0.0
        if answerable_main_question > 0.5:
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
                    memory_information = [f"- {item}" for item in memory_information if item]
                    memory_data = "\n".join(memory_information)
                    important_information += f"\t**Memory knowledge**\n{memory_data}\n----------\n"
                if external_information:
                    external_information = [f"- {item}" for item in external_information if item]
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
            children, explored_information = self.generate_final_answer_node()
        elif self.node_type in [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE]:
            children, explored_information = self.generate_subQA_node()
        elif self.node_type == NodeType.FINAL_ANSWER:
            # If the node is a final answer node, it has no children
            raise ValueError("Final answer nodes cannot have children.")
        else:
            raise ValueError(f"Invalid node type: {self.node_type}. Must be one of {list(NodeType)}.")
        children = list(set(children))  # Remove duplicates
        # remove the None children
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
        if len(children) == 0:
            print(f"No children generated for node type: {self.print_node()}")
            children = []
        return children, new_memory
    
    def find_children(self, rollout_id: Optional[int] = None):
        if self.children:
            return self.children
        children, new_memory = self.generate_children()
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





