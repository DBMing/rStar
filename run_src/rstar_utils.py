# Licensed under the MIT license.

from enum import Enum, unique
import re
import math
from typing import Dict, Tuple
from colorama import Fore, Style
import math


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    ONE_STEP = "ONE_STEP"
    
import time

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time() 
        print(f"{func.__name__} runs: {end_time - start_time:.6f} 秒")
        return result
    return wrapper


def reach_terminal_ost_step(ost_step: str):
    assert ost_step is not None
    last_step = ost_step.lower()
    
    code_indicators = [
            # "<action 3>",
            "```python"
        ]
        
    return any(indicator in last_step for indicator in code_indicators)


def print_tree_from_root(mcts_searcher, rollout_id, root_node, chosen_node=None, file=None):
    color_print = False if file else True

    def my_print(text):
        if file:
            file.write(text + "\n")
        else:
            print(text)

    def print_tree(parent_node, node, file, rollout_id):
        to_print = ""

        num_indent = 4
        dash = "-" * num_indent * node.depth
        space = " " * num_indent * node.depth

        attributes = f"Q: {round(mcts_searcher.Q[node], 2)}" + "; " + f"N: {mcts_searcher.N[node]}" + "; "
        attributes += f"V: {round(node.node_value, 2)}" if node.node_value is not None else "V: None"

        uct_value = "UCT: " + str(
            round(mcts_searcher._compute_uct(parent_node=parent_node, node=node, rollout_id=rollout_id), 2)
        )
        attributes += "; " + uct_value

        solution_marker = "(T) " if node.is_valid_solution_node() else ""

        node_info = "[" + solution_marker + node.__str__() + ": " + attributes + "]"
        if chosen_node and node == chosen_node:
            node_info = "[" + node_info + "]"
        node_info += " "

        if color_print and node.is_valid_solution_node():
            node_details = Fore.RED + Style.BRIGHT + node_info + Fore.RESET + Style.RESET_ALL
        else:
            node_details = node_info

        if node.node_type is Node_Type.USER_QUESTION:
            node_details += f"User: {node.user_question}" + "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.ONE_STEP:
            node_details += f"OST: {node.ost_step}"

        to_print += dash + node_details

        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)

        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated one-step thought steps, next one-step thought step id"""
    last_tuple = list(solution_trace.items())[-1]
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1]
    assert "ost_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost_step"]) > 0:
        solution_trace_str = ""
        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += f"<Step_Begin>\n### Step {step_id}: " + step_text + "<Step_End>\n\n"
        return solution_trace_str, step_id + 1
    else:
        # no one-step thought step yet
        return "", 1


def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []
    TREE = {}

    def recursion(node):
        if root_node.depth in TREE:
            TREE[root_node.depth].append(root_node)  
        else:
            TREE[root_node.depth] = [root_node]  
        
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return
        
        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes, TREE


def find_best_solution(root_node, evaluator, enable_potential_score=False):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.SUBQUESTION:
            return node.subanswer
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(extract_solution_from_node(node))
        potential_answers_history = node.potential_answers_history  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(
                    evaluator.check_answers_equiv(dpa, model_answer) for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None
    )
    top_answer, top_completion, top_completion_id, top_confidence = evaluator.find_most_confident_answer(
        solutions, prior_weights
    )
    return top_answer, top_completion, top_confidence, solution_nodes[top_completion_id], solution_nodes


def ost_find_best_solution(
    root_node,
    evaluator,
):
    solution_nodes, TREE = find_valid_solution_nodes(root_node)
    
    bestv = -1
    best_node = None
    for solution_node in solution_nodes:
        if solution_node.node_value > bestv:
            bestv = solution_node.node_value
            best_node = solution_node
            
    return best_node, solution_nodes, TREE
        
def find_solution(root_node, solution_node, mcts_searcher):
    """
    Recursively traces back from the given solution node to the root node, 
    calculating the value of each node along the path.

    Parameters:
    solution_node (Node): The current solution node to start the backtrace from.
    mcts_searcher (MCTS): The MCTS searcher object used to access node visit counts and values.

    Returns:
    dict: A dictionary representing the complete solution, containing the node id, 
          OST step, step value, and edge information for each node in the path.
    """
    comlete_solution = {}
    
    def reback(node):
        """
        Recursively backtracks from the current node to the root node, 
        calculating the value for each node and updating the solution.

        Parameters:
        node (Node): The current node being processed in the backtrack.
        """
        if node.node_value is not None and mcts_searcher.N[node] != 0:
            value = node.node_value / mcts_searcher.N[node]
        else:
            value = 0
        if node.node_type is Node_Type.ONE_STEP:
            comlete_solution[node.depth] = {
                "node_id": node.id, 
                "ost_step": node.ost_step, 
                "step_value": value, 
                "edges": (node.parent.id, node.id)   # source_node_id -> target_node_id
            }
        else:
            comlete_solution[node.depth] = {
                "node_id": node.id, 
                "question": root_node.user_question, 
            }
        if node.node_type is Node_Type.USER_QUESTION:
            return
        
        reback(node.parent)
        
    reback(solution_node)
    return comlete_solution
        
