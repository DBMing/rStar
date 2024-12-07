CUDA_VISIBLE_DEVICES=3 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename test_one \
    --model_ckpt /mnt/sdc1/jiangming/Project/rStar/Qwen/Qwen2.5-Coder-7B \
    --note default \
    --num_rollouts 3 \
    --verbose \
    --max_depth_allowed 8 