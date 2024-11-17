import sys
from types import SimpleNamespace

sys.path.append(".")

from models.IO_System import IO_System
from models.vLLM_API import load_vLLM_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = SimpleNamespace(
    note="default",
    api="vllm",
    seed=42,
    verbose=True,
    wandb_mode="disabled",
    model_ckpt="deepseek-ai/deepseek-coder-1.3b-instruct",
    model_parallel=False,
    half_precision=False,
    max_tokens=1024,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    num_beams=1,
    max_num_worker=3,
    test_batch_size=1,
    tensor_parallel_size=1,
    prompts_root="prompts",
    data_root="data",
    dataset_name="MBPPPLUS",
    test_json_filename="test_some",
    start_idx=0,
    run_outputs_root="run_outputs",
    eval_outputs_root="eval_outputs",
    num_rollouts=8,
    num_subquestions=3,
    num_votes=10,
    max_depth_allowed=12,
    mcts_discount_factor=1.0,
    mcts_exploration_weight=2.0,
    mcts_weight_scheduler="const",
    mcts_num_last_votes=32,
    save_tree=True,
    num_a1_steps=3,
    disable_a1=False,
    modify_prompts_for_rephrasing=False,
    disable_a5=False,
    enable_potential_score=False,
    fewshot_cot_prompt_path="prompts/MBPPPLUS/fewshot_cot/fewshot_cot_prompt.txt",
    fewshot_cot_config_path="prompts/MBPPPLUS/fewshot_cot/fewshot_cot_config.json",
    fewshot_ost_prompt_path="prompts/MBPPPLUS/fewshot_ost/fewshot_ost_prompt.txt",
    fewshot_ost_config_path="prompts/MBPPPLUS/fewshot_ost/fewshot_ost_config.json",
    decompose_template_path="prompts/MBPPPLUS/decompose/decompose_template.json",
    decompose_prompt_path="prompts/MBPPPLUS/decompose/decompose_prompt.txt",
    rephrasing_prompt_template_path="prompts/MBPPPLUS/rephrasing_prompt_template.txt",
    fewshot_cot_prompt_rephrased_path="prompts/MBPPPLUS/fewshot_cot/fewshot_cot_prompt.txt",
    fewshot_ost_prompt_rephrased_path="prompts/MBPPPLUS/fewshot_ost/fewshot_ost_prompt.txt",
    decompose_prompt_rephrased_path="prompts/MBPPPLUS/decompose/decompose_prompt.txt",
    run_outputs_dir="run_outputs/MBPPPLUS/Mistral-7B-v0.1/2024-10-29_19-17-33---[default]",
    answer_sheets_dir="run_outputs/MBPPPLUS/Mistral-7B-v0.1/2024-10-29_19-17-33---[default]/answer_sheets",
    cuda_0="NVIDIA GeForce RTX 3090"
)


tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)

io = IO_System(args, tokenizer, model)

# prompt = "You are an expert Python programmer, and here is your task: {}"
problem = "Write a python function to calculate the sum of two numbers"
# problem = "hello, what is the meaning of life?"

io_input = problem
print(io_input)

ans = io.generate(
            model_input=io_input, max_tokens=256, num_return=1, stop_tokens=["[Step Over]"]
            # model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=["[Step Over]"]
        )
print(ans)