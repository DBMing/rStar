CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename train_1000_v2 \
    --api gpt3.5-turbo \
    --model_ckpt gpt-4o-mini \
    --note default \
    --num_rollouts 3 \
    --verbose \
    --save_tree \
    --max_depth_allowed 8