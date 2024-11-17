CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name MBPPPLUS \
    --test_json_filename test_some \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --note default \
    --num_rollouts 8 \
    --verbose \
    --save_tree \
    --max_depth_allowed 12