# -*- coding: utf-8 -*-
# ===================================================================
# Training Settings
# ===================================================================
SOURCE_DIR = "./dataset/images_a"
TARGET_DIR = "./dataset/images_b"
GPU = 0
IMAGE_SIZE = 256

SAVE_DIR_BASE = './results'
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
    'content': 1.0, 
}

TRAINING_USE_EMA = True
RESUME_CHECKPOINT = None

# ===================================================================
# Inference Settings
# ===================================================================
INFERENCE_INPUT_DIR = './dataset/target/Tomato_healthy'
INFERENCE_TARGET_DIR = './dataset/images_b'
INFERENCE_CHECKPOINT_DIR = './results/exp_name/epoch_200'
INFERENCE_OUTPUT_DIR = './output/exp_name'
INFERENCE_DIRECTION = 'AtoB'

INFERENCE_USE_EMA = True

# The path to best-performing StyleCodeVAE model checkpoint.
INFERENCE_VAE_CHECKPOINT = './results/stylecodevae/checkpoints/stylecodevae_beta8.0_ld2.pth'
# The latent dimension used by the VAE model.
INFERENCE_VAE_LATENT_DIM = 2
# The style dimension of the SI-CycleGAN's encoder output.
INFERENCE_STYLE_DIM = 256

# ===================================================================
# Style Extracting Mode
# choices=['average', 'random', 'interpolate', 'noise']
# ===================================================================
INFERENCE_STYLE_MODE = 'interpolate'
INFERENCE_NOISE_LEVEL = 0.1

# ===================================================================
# Metrics Settings
# ===================================================================
METRICS_INPUT_DIR = './output/exp_name'
METRICS_TARGET_DIR = './dataset/all'

# ===================================================================
# StyleCodeVAE Experiment Settings
# ===================================================================
BETA_VALUES = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
LATENT_DIM = [2, 4, 8, 16, 32, 64, 128]

STYLE_ENCODER_CHECKPOINT = './results/stylecodevae/checkpoints/epoch_180'
SV_SAVE_DIR_BASE = './results/stylecodevae/checkpoints'

SV_NUM_EPOCHS = 10000