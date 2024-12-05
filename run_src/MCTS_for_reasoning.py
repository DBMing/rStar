# Licensed under the MIT license.

import sys

sys.path.append(".")

import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy

try:
    from rapidfuzz import fuzz, process
except:
    pass

from models.IO_System import IO_System
from common.utils import read_txt, read_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from run_src.rstar_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_ost_step,
    concat_subqs_and_subas,
    concat_ost_steps,
    concat_subqs_subas_as_ost_steps,
    make_hint,
    make_response_prefix,
    split_user_question,
    print_tree_from_root,
    find_valid_solution_nodes,
    find_best_solution,
    ost_find_best_solution,
    stochastic_find_best_solution,
    find_solution,
    time_decorator,
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator

        self.num_a1_steps = args.num_a1_steps
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score

        self.mcts_num_last_votes = args.mcts_num_last_votes

        

        if not args.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
            self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

    
    def _get_pass_code(self, io_output_list: List[str], user_question: str) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, passed_full_completion, _, confidence = self.evaluator.find_pass_code(io_output_list, user_question)
            assert confidence >= 0

        return passed_full_completion, confidence
    
    @time_decorator
    def _get_TACO_code(self, io_output_list: List[str], test_case: dict, solution_trace: Dict[int, Dict[str, str]],) -> Tuple[str, float]:
        assert len(io_output_list) > 0


        _, passed_full_completion, confidence, solution_trace_ = self.evaluator.find_TACO_code(io_output_list, test_case, solution_trace)
        assert confidence >= 0

        return passed_full_completion, confidence, solution_trace_

    def generate_ost_step(
        self,
        user_question: str,
        test_case: dict,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        ost_step_list = []
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples=self.fewshot_ost_prompt if not paraphrased else self.fewshot_ost_prompt_rephrased,
                instruction=user_question,
            )
            + existing_ost_steps
            + f"Step {next_ost_step_id}:"
        )
        # print(io_input)
        # io_input = {"role": "user", "content": io_input}
        # print(io_input)
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=5120, num_return=self.num_a1_steps, stop_tokens=["\\n\\n", "Step"]
        )
        ost_step_list = [io_output.strip() for io_output in io_output_list]

        last_ost_step = []
        value_list = []
        completion_confidence_list = []
        reach_last_step_flag = False
        # have_terminal_ost_step = False
        for ost_step in ost_step_list:
            if reach_terminal_ost_step(ost_step):
                reach_last_step_flag = True
                passed_full_completion, confidence, solution_trace_with_last_step = self._get_TACO_code([ost_step], test_case, solution_trace)
                completion_confidence_list.append((passed_full_completion, confidence))
            else:
                last_ost_step.append(ost_step)
                value_list.append(None)
        
        if reach_last_step_flag == True:
            last_ost_step.clear()
            value_list.clear()
            completion_confidence_list.sort(key=lambda x: x[1], reverse=True)
            best_passing_completion, highest_confidence = completion_confidence_list[0]
            
            for _, confidence in completion_confidence_list:
                print(confidence)
            return [best_passing_completion], [highest_confidence], [None]
        else:
            potential_answers_list: List[List[str]] = []
            print(value_list)
            if value_list.count(None) != 0 and value_list.count(None) != len(value_list):
                for idx, value in enumerate(value_list):
                    if value is not None:
                        number = value
                        corresponding_step = last_ost_step[idx]
                        break
                value_list = [number]
                last_ost_step = [corresponding_step]
            potential_answers_list = [None] * len(value_list)
            return last_ost_step, value_list, potential_answers_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        expected_answer: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        # ---------------------------------------
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
        test_case: dict = None,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                print(node_value)
                assert node_value >= 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, user_question, expected_answer, max_depth_allowed, disable_a1]
                )
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        user_question,
                        expected_answer, 
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.ost_step = ost_step
        self.test_case = test_case

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
            self.enable_potential_score = enable_potential_score
            self.test_case = test_case
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1
            self.enable_potential_score = parent.enable_potential_score
            self.test_case = parent.test_case

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased


        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}, "ost_step_value": {}}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.OST_STEP:
                assert "ost_step" in self.solution_trace[0].keys()
                self.solution_trace[0]["ost_step"][self.ost_step_counter] = ost_step
                self.solution_trace[0]["ost_step_value"][self.ost_step_counter] = node_value

        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.OST_STEP: "TS",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):

        def do_action_generate_ost_step():
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, value_list, potential_answers_list = self.generator.generate_ost_step(
                user_question=self.user_question,
                test_case = self.test_case,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for ost_step, value, potential_answers in zip(ost_step_list, value_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        node_value=value,
                        ost_step=ost_step,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

       
        elif self.node_type is Node_Type.OST_STEP:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            # do_action_generate_direct_answers()

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0


def search_for_answers(args, user_question: str, question_id: int, gt_answer: str, generator: Generator, test_case: dict):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        user_question=user_question,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        disable_a1=args.disable_a1,
        enable_potential_score=args.enable_potential_score,
        test_case=test_case,
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)
        jss = {"trace": rollout_node.solution_trace, "rollout_id": rollout_node.rollout_id, "value": rollout_node.node_value}
    
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - rollout Solutions.json"), "a") as f:
            json.dump(jss, f)
            f.write(',')
        
    ost_best_node, ost_all_solution_nodes, TREE = ost_find_best_solution(root_node, generator.evaluator)
        
    complete_road = []
    
    for solution_node in ost_all_solution_nodes:
        complete_road_json = find_solution(solution_node, mcts_searcher)
        complete_road.append(complete_road_json)
            
            
    bestv = -1
    ost_best_node = None
    for rollout_node in model_rollout_nodes:
        if rollout_node.node_value is not None:
            if rollout_node.node_value > bestv:
                bestv = rollout_node.node_value
                ost_best_node = rollout_node
    
        
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Complete Solutions.json"), "w", encoding="utf-8") as f:
        json.dump(complete_road, f, ensure_ascii=False, indent=4)
    #! record final traces
    js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id, "parent_id": node.parent.id, "value": node.node_value} for node in ost_all_solution_nodes]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
        json.dump(js, f)

    js2 = [{"trace": node.solution_trace, "rollout_id": i, "value": node.node_value} for i, node in enumerate(model_rollout_nodes)]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
        json.dump(js2, f)
    
    if ost_best_node is not None:
        js3 = {"trace": ost_best_node.solution_trace, "rollout_id": ost_best_node.rollout_id, "value": ost_best_node.node_value}
    
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Best Solutions.json"), "w") as f:
            json.dump(js3, f)
            
   

    if args.enable_potential_score:
        js = [node.potential_answers_history for node in ost_all_solution_nodes]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Potentials.json"), "w") as f:
            json.dump(js, f) 

    
    # if TREE is not None:
    #     for k, v in TREE.items():
    #         js2 = [{"depth": k, "node_id": node.id, "ost_step": node.ost_step} for i, node in enumerate(model_rollout_nodes)]
    #     with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - TREE.json"), "w") as f:
    #         json.dump(TREE, f)
        # with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - TREE.json"), "w") as f:
        #     json.dump(TREE, f)

    return model_solutions, i, model_all_solutions
