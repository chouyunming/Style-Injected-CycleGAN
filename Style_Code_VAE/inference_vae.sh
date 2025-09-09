#!/bin/bash

# This script automates running inference for a StyleCycleGAN model with VAE style mode.
# It iterates through different VAE checkpoints and applies VAE-based style generation.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE_PATH="./stylecyclegan/config.py"

# --- Function to safely read a variable from config.py ---
read_config_var() {
    local var_name=$1
    grep "$var_name" "$CONFIG_FILE_PATH" | awk -F'=' '{print $2}' | tr -d " ',\r"
}

# --- Read Paths and Settings from Config ---
GAN_CHECKPOINT_DIR=$(read_config_var "INFERENCE_CHECKPOINT_DIR")
CONTENT_INPUT_DIR=$(read_config_var "INFERENCE_INPUT_DIR")
TARGET_DOMAIN_DIR=$(read_config_var "TARGET_DIR")
VAE_BASE_DIR=$(read_config_var "SV_SAVE_DIR_BASE")
VAE_STYLE_DIM=$(read_config_var "INFERENCE_STYLE_DIM")
BASE_OUTPUT_DIR="./stylecyclegan/output/vae_inference"

DIRECTION="AtoB"
GPU_ID=0
IMAGE_SIZE=256
USE_EMA=true

# --- Define the style mode ---
STYLE_MODE="vae"

echo "=============================================================="
echo "Starting StyleCycleGAN VAE Inference Script"
echo "Style mode: $STYLE_MODE"
echo "=============================================================="

# --- Validation ---
if [ ! -d "$VAE_BASE_DIR" ]; then
    echo "ERROR: VAE checkpoint directory not found: $VAE_BASE_DIR"
    exit 1
fi

if [ ! -d "$CONTENT_INPUT_DIR" ]; then
    echo "ERROR: Content input directory not found: $CONTENT_INPUT_DIR"
    exit 1
fi

echo "GAN Checkpoint: $GAN_CHECKPOINT_DIR"
echo "VAE Checkpoints Directory: $VAE_BASE_DIR"
echo "Content Input Directory: $CONTENT_INPUT_DIR"
echo "Target Domain Directory: $TARGET_DOMAIN_DIR"
echo "All outputs will be saved under: $BASE_OUTPUT_DIR"
echo "--------------------------------------------------------------"

# --- Main Loop - Iterate through VAE checkpoint files ---
for vae_checkpoint_path in "$VAE_BASE_DIR"/*.pth; do
    if [ ! -f "$vae_checkpoint_path" ]; then
        echo "Warning: No .pth files found in $VAE_BASE_DIR"
        continue
    fi
    
    # Parse beta and latent_dim from the filename
    vae_filename=$(basename "$vae_checkpoint_path")
    BETA=$(echo "$vae_filename" | sed -n 's/.*beta\([0-9.]*\)_ld.*/\1/p')
    LDIM=$(echo "$vae_filename" | sed -n 's/.*ld\([0-9]*\).*/\1/p')

    if [ -z "$BETA" ] || [ -z "$LDIM" ]; then
        echo "Warning: Could not parse beta/ldim from filename '${vae_filename}'. Skipping."
        continue
    fi
    
    echo "Processing VAE Checkpoint: ${vae_filename}"
    echo " -> Beta: $BETA, Latent Dim: $LDIM"
    
    # Create output directory for this specific VAE model
    output_dir="${BASE_OUTPUT_DIR}/beta${BETA}_ld${LDIM}"
    mkdir -p "$output_dir"
    
    echo " -> VAE Checkpoint: $vae_checkpoint_path"
    echo " -> Output Directory: $output_dir"
    
    CMD="python3 ./stylecyclegan/inference.py \
        --input_dir \"$CONTENT_INPUT_DIR\" \
        --target_domain_dir \"$TARGET_DOMAIN_DIR\" \
        --output_dir \"$output_dir\" \
        --checkpoint_dir \"$GAN_CHECKPOINT_DIR\" \
        --direction \"$DIRECTION\" \
        --gpu \"$GPU_ID\" \
        --image_size \"$IMAGE_SIZE\" \
        --use_ema \"$USE_EMA\" \
        --style_mode \"$STYLE_MODE\" \
        --vae_checkpoint_path \"$vae_checkpoint_path\" \
        --vae_latent_dim \"$LDIM\" \
        --vae_style_dim \"$VAE_STYLE_DIM\""
    
    eval $CMD
    echo "Finished processing ${vae_filename}. Results saved."
    echo ".............................................................."
done

echo "VAE inference tasks complete!"