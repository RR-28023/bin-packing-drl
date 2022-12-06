python src/main.py \
    --inference=True \
    --inference_data_path "" \
    --min_item_size $1 \
    --max_item_size $2 \
    --min_num_items $3 \
    --max_num_items $4 \
    --bin_size $5 \
    --agent_heuristic NF \
    --model_path ./experiments/models/DRL-NF_size_$1_$2_items_$3_$4_bin_$5.pkl
