#!/bin/bash

# This script automates the generation of sample images by using all trained
# StyleCodeVAE models to drive a StyleCycleGAN generator. This is essential
# for evaluating which VAE hyperparameters produce the best results.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE_PATH="./stylecyclegan/config.py"

# --- Function to safely read a variable from config.py ---
# This function is now more robust and removes quotes, spaces, commas,
# and carriage return characters (\r) to prevent issues with file formats.
read_config_var() {
    local var_name=$1
    grep "$var_name" "$CONFIG_FILE_PATH" | awk -F'=' '{print $2}' | tr -d " ',\r"
}

# --- Read Paths and Settings from Config ---
GAN_CHECKPOINT_DIR=$(read_config_var "STYLE_ENCODER_CHECKPOINT")
CONTENT_INPUT_DIR=$(read_config_var "INFERENCE_INPUT_DIR")
VAE_BASE_DIR=$(read_config_var "SV_SAVE_DIR_BASE")
VAE_STYLE_DIM=$(read_config_var "INFERENCE_STYLE_DIM")
BASE_OUTPUT_DIR="./stylecyclegan/output/stylecodevae"

# --- General Settings ---
NUM_IMAGES=200
GPU_ID=0

echo "======================================================================"
echo "Starting Batch Image Generation from StyleCodeVAE Models"
echo "======================================================================"

# --- Validation ---
# The check for GAN_CHECKPOINT_DIR is commented out because it's a file prefix, not a directory.
# if [ ! -d "$GAN_CHECKPOINT_DIR" ]; then
#     echo "ERROR: StyleCycleGAN checkpoint directory not found at path specified by 'STYLE_ENCODER_CHECKPOINT' in config: $GAN_CHECKPOINT_DIR"
#     exit 1
# fi

# This check is valid and ensures the VAE models directory exists.
if [ ! -d "$VAE_BASE_DIR" ]; then
    echo "ERROR: Base directory for VAE models not found at path specified by 'SV_SAVE_DIR_BASE' in config: $VAE_BASE_DIR"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"
echo "GAN Checkpoint Prefix: $GAN_CHECKPOINT_DIR"
echo "Scanning for VAE models in: $VAE_BASE_DIR"
echo "All generated images will be saved under: $BASE_OUTPUT_DIR"
echo "----------------------------------------------------------------------"

# --- Main Loop ---
# Loop through all found .pth files in the VAE directory
for vae_checkpoint_path in "$VAE_BASE_DIR"/*.pth; do
    if [ ! -f "$vae_checkpoint_path" ]; then
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

    echo "Processing VAE Model: ${vae_filename}"
    
    # Create a specific, descriptive output directory
    output_dir="${BASE_OUTPUT_DIR}/beta${BETA}_ld${LDIM}"
    mkdir -p "$output_dir"
    
    echo " -> VAE Checkpoint: $vae_checkpoint_path"
    echo " -> Latent Dim: $LDIM"
    echo " -> Output Directory: $output_dir"
    
    # Build and execute the command for generate_images.py
    # Note: The script name was corrected from stylecyclegangenerate_images.py to generate_images.py
    CMD="python3 ./stylecyclegan/generate_images.py \
        --gan_checkpoint_dir \"$GAN_CHECKPOINT_DIR\" \
        --vae_checkpoint_path \"$vae_checkpoint_path\" \
        --content_dir \"$CONTENT_INPUT_DIR\" \
        --output_dir \"$output_dir\" \
        --vae_latent_dim \"$LDIM\" \
        --vae_style_dim \"$VAE_STYLE_DIM\" \
        --num_images \"$NUM_IMAGES\" \
        --gpu \"$GPU_ID\""
    
    eval $CMD

    echo "Finished processing ${vae_filename}. Results saved."
    echo "......................................................................"
done

echo "All VAE evaluation image generation tasks are complete!"