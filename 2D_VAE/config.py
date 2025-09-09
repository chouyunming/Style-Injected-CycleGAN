# ===================================================================
# VAE Experiment Settings
# ===================================================================
BETA_VALUES = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
LATENT_DIM = [2, 4, 8, 16, 32, 64, 128]

SOURCE_DIR = "./dataset/images_a"
TARGET_DIR = "./dataset/images_b"

SAVE_DIR_BASE = './results'
OUTPUT_DIR_BASE = './output'

NUM_EPOCHS = 1000
BATCH_SIZE = 4
IMAGE_SIZE = 256