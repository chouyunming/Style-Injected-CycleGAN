#!/bin/bash

# This script automates a grid search for the FewShotVAE.py script
# by iterating through different beta and latent_dim values specified
# in the config.py file.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "================================================="
echo "Starting VAE Hyperparameter Grid Search Script"
echo "================================================="

# --- Step 1: Read configuration from config.py ---

# This function safely extracts a list of numbers from the config file
extract_list() {
    local list_name=$1
    # --- MODIFIED: Point to the config file inside the stylecyclegan directory ---
    local config_file="./config.py"
    
    if [ ! -f "${config_file}" ]; then
        echo "Error: Config file not found at ${config_file}"
        exit 1
    fi

    if ! grep -q "${list_name}" "${config_file}"; then
        echo "Error: Could not find '${list_name}' in ${config_file}."
        exit 1
    fi
    
    grep "${list_name}" "${config_file}" | sed 's/.*\[\(.*\)\].*/\1/' | tr -d ' ' | sed 's/,/ /g'
}

BETA_LIST=$(extract_list "BETA_VALUES")
LATENT_DIM_LIST=$(extract_list "LATENT_DIM")

if [ -z "$BETA_LIST" ] || [ -z "$LATENT_DIM_LIST" ]; then
    echo "Error: Failed to extract BETA_VALUES or LATENT_DIM from config.py."
    exit 1
fi

echo "Found Beta values to test: $BETA_LIST"
echo "Found Latent Dimensions to test: $LATENT_DIM_LIST"
echo "----------------------------------------"


# --- Step 2: Loop through all combinations and run training ---

# Outer loop for Beta values
for BETA in $BETA_LIST
do
    # Inner loop for Latent Dimension values
    for LDIM in $LATENT_DIM_LIST
    do
        echo ""
        echo "*************************************************"
        echo "*** Training with Beta = $BETA and Latent Dim = $LDIM ***"
        echo "*************************************************"
        echo ""
        
        # Execute the Python script, passing the current beta and latent_dim as arguments
        python ./FewShotVAE.py --beta "$BETA" --latent_dim "$LDIM"
        
        echo ""
        echo "*** Finished training for Beta = $BETA, Latent Dim = $LDIM ***"
        echo "----------------------------------------"
    done
done

echo ""
echo "================================================="
echo "All training grid search runs completed."
echo "================================================="
