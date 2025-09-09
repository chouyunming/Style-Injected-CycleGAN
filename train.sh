#!/bin/bash

# --- GLOBAL SETTING ---
# Make sure to log in to W&B first: wandb login
SOURCE_DIR="./stylecyclegan/dataset/images_a"
TARGET_DIR="./stylecyclegan/dataset/images_b"
EPOCHS=200
BATCH_SIZE=4

# # ========================= EXPERIMENT 1 ==========================
# echo "--- Preparing Experiment 1: ---"

# EXP1_NAME="exp1_lrg2e4_lrd1e4_gan5_cycle7_identity1_style3_content3"
# EXP1_LR_G=2e-4
# EXP1_LR_D=1e-4
# EXP1_WEIGHTS='{"gan": 5.0, "cycle": 7.0, "identity": 1.0, "style": 3.0, "content": 3.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP1_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP1_LR_G \
#   --lr_d $EXP1_LR_D \
#   --loss_weights "$EXP1_WEIGHTS" \
#   --wandb

# echo "--- Experiment 1 Finished ---"
# echo ""


# # ========================= EXPERIMENT 2 ==========================
# echo "--- Preparing Experiment 2: ---"

# EXP2_NAME="exp2_lrg2e4_lrd1e4_gan5_cycle7_identity5_style3_content3"
# EXP2_LR_G=2e-4
# EXP2_LR_D=1e-4
# EXP2_WEIGHTS='{"gan": 5.0, "cycle": 7.0, "identity": 5.0, "style": 3.0, "content": 3.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP2_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP2_LR_G \
#   --lr_d $EXP2_LR_D \
#   --loss_weights "$EXP2_WEIGHTS" \
#   --wandb

# echo "--- Experiment 2 Finished ---"
# echo ""

# ========================= EXPERIMENT 3 ==========================
# echo "--- Preparing Experiment 3: ---"

# EXP3_NAME="exp3_lrg2e4_lrd1e4_gan5_cycle7_identity5_style10_content3"
# EXP3_LR_G=2e-4
# EXP3_LR_D=1e-4
# EXP3_WEIGHTS='{"gan": 5.0, "cycle": 7.0, "identity": 5.0, "style": 10.0, "content": 3.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP3_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP3_LR_G \
#   --lr_d $EXP3_LR_D \
#   --loss_weights "$EXP3_WEIGHTS" \
#   --resume './stylecyclegan/results/exp3_lrg2e4_lrd1e4_gan5_cycle7_identity5_style10_content3/checkpoints/epoch_140' \
#   --wandb

# echo "--- Experiment 3 Finished ---"
# echo ""

# # ========================= EXPERIMENT 4 ==========================
# echo "--- Preparing Experiment 4: ---"

# EXP4_NAME="exp4_lrg2e4_lrd1e4_gan1_cycle1_identity1_style1_content1"
# EXP4_LR_G=2e-4
# EXP4_LR_D=1e-4
# EXP4_WEIGHTS='{"gan": 1.0, "cycle": 1.0, "identity": 1.0, "style": 1.0, "content": 1.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP4_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP4_LR_G \
#   --lr_d $EXP4_LR_D \
#   --loss_weights "$EXP4_WEIGHTS" \
#   --wandb

# echo "--- Experiment 4 Finished ---"
# echo ""

# # ========================= EXPERIMENT 5 ==========================
# echo "--- Preparing Experiment 5: ---"

# EXP5_NAME="exp5_nblock6_lrg2e4_lrd1e4_gan5_cycle7_identity1_style3_content3"
# EXP5_LR_G=2e-4
# EXP5_LR_D=1e-4
# EXP5_WEIGHTS='{"gan": 5.0, "cycle": 7.0, "identity": 1.0, "style": 3.0, "content": 3.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP5_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP5_LR_G \
#   --lr_d $EXP5_LR_D \
#   --loss_weights "$EXP5_WEIGHTS" \
#   --wandb

# echo "--- Experiment 5 Finished ---"
# echo ""

# # ========================= EXPERIMENT 6 ==========================
# echo "--- Preparing Experiment 6: ---"

# EXP6_NAME="exp6_lrg2e4_lrd1e4_gan1_cycle10_identity5_style1_content1"
# EXP6_LR_G=2e-4
# EXP6_LR_D=1e-4
# EXP6_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP6_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP6_LR_G \
#   --lr_d $EXP6_LR_D \
#   --loss_weights "$EXP6_WEIGHTS" \
#   --wandb

# echo "--- Experiment 6 Finished ---"
# echo ""

# # ========================= EXPERIMENT 7 ==========================
# echo "--- Preparing Experiment 7: ---"

# EXP7_NAME="exp7_lrg2e4_lrd1e4_gan1_cycle10_identity5_style1_content1"
# EXP7_LR_G=2e-4
# EXP7_LR_D=1e-4
# EXP7_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP7_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP7_LR_G \
#   --lr_d $EXP7_LR_D \
#   --loss_weights "$EXP7_WEIGHTS" \
#   --wandb

# echo "--- Experiment 7 Finished ---"
# echo ""

# ========================= EXPERIMENT 8 ==========================
echo "--- Preparing Experiment 8: ---"

EXP8_NAME="exp8_lrg2e4_lrd1e4_gan1_cycle10_identity5_style1_content1_skip0.1"
EXP8_LR_G=2e-4
EXP8_LR_D=1e-4
EXP8_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

python ./stylecyclegan/main.py \
  --exp_name "$EXP8_NAME" \
  --source_dir "$SOURCE_DIR" \
  --target_dir "$TARGET_DIR" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr_g $EXP8_LR_G \
  --lr_d $EXP8_LR_D \
  --loss_weights "$EXP8_WEIGHTS" \
  --skip_connection True \
  --wandb

echo "--- Experiment 8 Finished ---"
echo ""

# # ========================= EXPERIMENT 9 ==========================
# echo "--- Preparing Experiment 9: ---"

# EXP9_NAME="exp9_lrg2e4_lrd1e4_gan1_cycle10_identity5_style1_content1"
# EXP9_LR_G=2e-4
# EXP9_LR_D=1e-4
# EXP9_WEIGHTS='{"gan": 1.0, "cycle": 10.0, "identity": 5.0, "style": 1.0, "content": 1.0}'

# python ./stylecyclegan/main.py \
#   --exp_name "$EXP9_NAME" \
#   --source_dir "$SOURCE_DIR" \
#   --target_dir "$TARGET_DIR" \
#   --epochs $EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --lr_g $EXP9_LR_G \
#   --lr_d $EXP9_LR_D \
#   --loss_weights "$EXP9_WEIGHTS" \
#   --wandb

# echo "--- Experiment 9 Finished ---"
# echo ""

echo "All experiments completed."