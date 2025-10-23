#!/bin/bash

# This script automates running inference for a StyleCycleGAN model.
# It iterates through different domains and GAN checkpoints for each domain.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SAVE_DIR_BASE="./stylecyclegan"
CONFIG_FILE_PATH="./stylecyclegan/config.py"
CONTENT_INPUT_DIR="./experiments/plant_village_raw/synthetic_target/Tomato_healthy"
DIRECTION="AtoB"
GPU_ID=0
IMAGE_SIZE=256
USE_EMA=true

# --- Define the domains to process ---
DOMAINS=(
    "Tomato__Tomato_mosaic_virus"
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
    "Tomato_Early_blight"
    "Tomato_Late_blight"
    "Tomato_Leaf_Mold"
    "Tomato_Septoria_leaf_spot"
    "Tomato_Spider_mites_Two_spotted_spider_mite"
)

# --- Define the single style mode to run ---
# Available options: average, random, interpolate, noise, vae
STYLE_MODE="interpolate"

echo "=============================================================="
echo "Starting StyleCycleGAN Batch Inference Script"
echo "Style mode: $STYLE_MODE"
echo "Domains to process: ${#DOMAINS[@]}"
echo "=============================================================="

# --- Main Loop for Domains ---
for domain_name in "${DOMAINS[@]}"; do
    echo ""
    echo "======== Processing Domain: $domain_name ========"
    
    EXPERIMENT_NAME="exp_Alan_${domain_name}"
    BASE_CHECKPOINT_DIR="${SAVE_DIR_BASE}/results/${EXPERIMENT_NAME}/checkpoints"
    BASE_OUTPUT_DIR="${SAVE_DIR_BASE}/output/${EXPERIMENT_NAME}"
    TARGET_DOMAIN_DIR="./experiments/exp_Alan/ref/${domain_name}"
    
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Checkpoint dir: $BASE_CHECKPOINT_DIR"
    echo "Output dir: $BASE_OUTPUT_DIR"
    echo "Target domain dir: $TARGET_DOMAIN_DIR"
    echo "--------------------------------------------------------------"
    
    # --- Process all checkpoints for this domain ---
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
        echo "Finished processing checkpoint ${epoch_dir_name}."
        echo ".............................................................."
    done
    
    echo "Completed domain: $domain_name"
    echo "--------------------------------------------------------------"
done

echo ""
echo "=============================================================="
echo "Batch inference task complete for all domains!"
echo "=============================================================="