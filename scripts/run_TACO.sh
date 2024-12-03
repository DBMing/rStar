CUDA_VISIBLE_DEVICES=4 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename test_one \
    --model_ckpt /mnt/sdc1/jiangming/Project/rStar/mistralai/Mistral-7B-v0.1 \
    --note default \
    --num_rollouts 3 \
    --verbose \
    --save_tree \
    --max_depth_allowed 8 