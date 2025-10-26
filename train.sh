#!/bin/bash

# --- GLOBAL SETTING ---
# Make sure to log in to W&B first: wandb login
SOURCE_DIR="./dataset/images_a"
TARGET_DIR="./dataset/images_b"
EPOCHS=200
BATCH_SIZE=4

# ========================= EXPERIMENT 1 ==========================
echo "--- Preparing Experiment 1: ---"

EXP1_NAME="base_experiment"
EXP1_LR_G=2e-4
EXP1_LR_D=1e-4
EXP1_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

python ./main.py \
  --exp_name "$EXP1_NAME" \
  --source_dir "$SOURCE_DIR" \
  --target_dir "$TARGET_DIR" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr_g $EXP1_LR_G \
  --lr_d $EXP1_LR_D \
  --loss_weights "$EXP1_WEIGHTS" 

echo "--- Experiment 1 Finished ---"
echo ""

echo "All experiments completed."