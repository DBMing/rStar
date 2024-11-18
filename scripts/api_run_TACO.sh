CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename test_one \
    --api gpt3.5-turbo \
    --model_ckpt gpt-4o-mini \
    --note default \
    --num_rollouts 12 \
    --verbose \
    --max_depth_allowed 10