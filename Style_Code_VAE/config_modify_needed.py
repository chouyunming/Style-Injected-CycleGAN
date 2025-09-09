# -*- coding: utf-8 -*-
# ===================================================================
# StyleCycleGAN Training Settings
# ===================================================================
SOURCE_DIR = "./stylecyclegan/dataset/images_a"
TARGET_DIR = "./stylecyclegan/dataset/images_b"
GPU = 0
IMAGE_SIZE = 256

SAVE_DIR_BASE = './stylecyclegan/results'
EXPERIMENT_NAME = 'base_experiment'

NUM_EPOCHS = 200
BATCH_SIZE = 4
SAVE_FREQ = 50

N_RESIDUAL_BLOCKS = 8

LEARNING_RATE_G = 2e-4  
LEARNING_RATE_D = 1e-4  

LOSS_WEIGHTS = {
    'gan': 1.0, 
    'cycle': 10.0, 
    'identity': 5.0, 
    'style': 1.0, 
    'content': 1.0
}

TRAINING_USE_EMA = True

RESUME_CHECKPOINT = None

# ===================================================================
# StyleCycleGAN Inference Settings
# ===================================================================
INFERENCE_INPUT_DIR = './experiments/plant_village_raw/synthetic_target/Tomato_healthy'
INFERENCE_CHECKPOINT_DIR = './stylecyclegan/results/the_first_test/checkpoints/epoch_180'
INFERENCE_OUTPUT_DIR = './stylecyclegan/output/style_mode_the_first_test/stylecodevae_beta8.0_ld2'
INFERENCE_DIRECTION = 'AtoB'

INFERENCE_USE_EMA = True

# The path to best-performing StyleCodeVAE model checkpoint.
INFERENCE_VAE_CHECKPOINT = './stylecyclegan/results/stylecodevae/checkpoints/stylecodevae_beta8.0_ld2.pth'
# The latent dimension used by the VAE model.
INFERENCE_VAE_LATENT_DIM = 2
# The style dimension of the StyleCycleGAN's encoder output.
INFERENCE_STYLE_DIM = 256

# ===================================================================
# Style Extracting Mode
# choices=['average', 'random', 'interpolate', 'noise']
# ===================================================================
INFERENCE_STYLE_MODE = 'noise'
INFERENCE_NOISE_LEVEL = 0.1

# ===================================================================
# Metrics Per Epoch Experiment Settings
# ===================================================================
METRICS_INPUT_DIR = './stylecyclegan/output/vae_inference'
METRICS_TARGET_DIR = './experiments/plant_village_raw/train/Tomato_Bacterial_spot'

# ===================================================================
# StyleCodeVAE Experiment Settings
# ===================================================================
BETA_VALUES = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
LATENT_DIM = [2, 4, 8, 16, 32, 64, 128]

STYLE_ENCODER_CHECKPOINT = './stylecyclegan/results/the_first_test/checkpoints/epoch_180'
SV_SAVE_DIR_BASE = './stylecyclegan/results/stylecodevae/checkpoints'

SV_NUM_EPOCHS = 10000