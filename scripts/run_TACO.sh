CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename test_one \
    --model_ckpt Qwen/Qwen2.5-Coder-32B \
    --note default \
    --num_rollouts 3 \
    --verbose \
    --save_tree \
    --max_depth_allowed 8 