#!/bin/bash

# This script automates running inference for a StyleCycleGAN model.
# It iterates through different GAN checkpoints and applies a single style mode.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SAVE_DIR_BASE="./stylecyclegan"
EXPERIMENT_NAME="exp7_lrg2e4_lrd1e4_gan1_cycle10_identity5_style1_content1"
CONFIG_FILE_PATH="./stylecyclegan/config.py"

BASE_CHECKPOINT_DIR="${SAVE_DIR_BASE}/results/${EXPERIMENT_NAME}/checkpoints"
BASE_OUTPUT_DIR="${SAVE_DIR_BASE}/output/${EXPERIMENT_NAME}"
CONTENT_INPUT_DIR="./experiments/plant_village_raw/synthetic_target/Tomato_healthy"
TARGET_DOMAIN_DIR="./experiments/plant_village_raw/train/Tomato_Bacterial_spot_few"

DIRECTION="AtoB"
GPU_ID=0
IMAGE_SIZE=256
USE_EMA=true

# --- Define the single style mode to run ---
# Available options: average, random, interpolate, noise, vae
STYLE_MODE="interpolate"

echo "=============================================================="
echo "Starting StyleCycleGAN Single Mode Inference Script"
echo "Style mode: $STYLE_MODE"
echo "=============================================================="

# --- Validation ---
if [ ! -d "$BASE_CHECKPOINT_DIR" ]; then
    echo "ERROR: Base checkpoint directory not found: $BASE_CHECKPOINT_DIR"
    exit 1
fi

echo "All outputs will be saved under: $BASE_OUTPUT_DIR"
echo "--------------------------------------------------------------"

# --- Main Loop ---
# FIX: Added '/*' to the end of the directory path.
# This makes the loop iterate over all items (files and directories) inside BASE_CHECKPOINT_DIR.
for checkpoint_path in "$BASE_CHECKPOINT_DIR"/*; do
    # Check if the item is a directory. If not, skip it.
    if [ ! -d "$checkpoint_path" ]; then
        continue
    fi
    
    # Get the name of the subdirectory (e.g., "epoch_10")
    epoch_dir_name=$(basename "$checkpoint_path")
    
    echo "Processing Checkpoint: ${epoch_dir_name} | Style Mode: ${STYLE_MODE}"
    
    # Create a corresponding output directory
    output_dir="${BASE_OUTPUT_DIR}/${STYLE_MODE}/${epoch_dir_name}"
    mkdir -p "$output_dir"
    
    echo " -> GAN Checkpoint: $checkpoint_path"
    echo " -> Output Directory: $output_dir"
    
    # Construct and execute the python command
    CMD="python3 ./stylecyclegan/inference.py \
        --input_dir \"$CONTENT_INPUT_DIR\" \
        --target_domain_dir \"$TARGET_DOMAIN_DIR\" \
        --output_dir \"$output_dir\" \
        --checkpoint_dir \"$checkpoint_path\" \
        --direction \"$DIRECTION\" \
        --gpu \"$GPU_ID\" \
        --image_size \"$IMAGE_SIZE\" \
        --use_ema \"$USE_EMA\" \
        --style_mode \"$STYLE_MODE\""
    
    eval $CMD
    echo "Finished processing. Results saved."
    echo ".............................................................."
    echo "--------------------------------------------------------------"
done

echo "Single mode inference task complete!"