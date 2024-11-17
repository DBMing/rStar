CUDA_VISIBLE_DEVICES=0 python run_src/do_generate_gsm.py \
    --dataset_name GSM8K \
    --test_json_filename test_all \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --note default \
    --num_rollouts 16
