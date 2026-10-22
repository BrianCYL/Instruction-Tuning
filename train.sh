if [ ${1} == "gemma" ]; then
    model_path="zake7749/gemma-2-2b-it-chinese-kyara-dpo"
elif [ ${1} == "llama" ]; then
    model_path="yentinglin/Taiwan-LLM-7B-v2.0-chat"
else
    echo "Invalid model name"
fi

python train.py \
    --model_path $model_path \
    --train_file_dir ./data/train.json \
    --test_file_dir ./data/public_test.json \
    --max_input_length 512 \
    --train_batch_size 4 \
    --n_epochs 3 \
    --output_dir ./checkpoints_${1}\
    --lr 2e-4 \
    --gradient_accumulation_steps 8 



