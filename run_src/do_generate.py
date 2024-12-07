# Licensed under the MIT license.

import sys


print("args: ", sys.argv)

import os, json, time
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(".")

from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from MCTS_for_reasoning import Generator, search_for_answers
from eval_src.Evaluator import *


def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)
    elif args.api == "OpenAI":
        from models.OpenAI_API import load_OpenAI_model

        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    generator = Generator(args, tokenizer, model, evaluator)

    num_tested = 0
    start_time = time.time()

    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < args.start_idx or i >= args.end_idx:
            continue
        st_time = time.time()
        problem_id, problem, test_case, difficulty = i, data_item["question"],data_item["input_output"], data_item["difficulty"]
        gt_solution = data_item["solutions"][0] if len(data_item["solutions"]) > 0 else None

        js = {
            "id": problem_id,
            "problem": problem,
            "gold_solution": gt_solution,
            "test_case": test_case,
            "difficulty": difficulty,
        }

        
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args, user_question=problem, question_id=i, difficulty=difficulty, generator=generator, test_case = test_case
        )
        
        num_tested += 1

        

        with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
            f.write(
                f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
            )
            f.write(
                f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
            )
        ed_time = time.time()
        js["time_taken"] = f"{ed_time-st_time:.2f}s"
        
        with open(os.path.join(args.answer_sheets_dir, f"Question {i:04d} - Answer.json"), "w") as f:
            json.dump(js, f)
        print(f"==> Time taken for this question: {ed_time-st_time:.2f}s")

    end_time = time.time()

    print(f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}")
    print(f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}")
    print(f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s")

    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "w") as f:
        f.write(f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n")
        f.write(
            f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
        )
        f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")


if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")

    # Action1: Propose an one-step thought.
    parser.add_argument("--num_sampling", type=int, default=3)


    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")

    #! -------------------------------------------------------------------------------

    args = parser.parse_args()

    if args.mcts_num_last_votes is None:
        args.mcts_num_last_votes = 32

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)


    args.examples_txt_path = os.path.join(prompts_dir, "examples.txt")
    args.prompt_config_path = os.path.join(prompts_dir, "prompt.json")
    
    
    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
