CUDA_VISIBLE_DEVICES=0,1 python run_src/do_generate.py \
    --dataset_name TACO \
    --test_json_filename test_one \
    --model_ckpt /root/shared-nvme/Qwen2.5-Coder-14B-Instruct \
    --note default \
    --num_rollouts 6 \
    --verbose \
    --max_depth_allowed 8 \
    --tensor_parallel_size 2 \
    --run_outputs_dir test_Q_10_rollout_6