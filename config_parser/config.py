from config_parser.base import Config, PathConfig

_CONFIG_PATH = "config.json"
CONFIG = Config(_CONFIG_PATH)
PATHS = PathConfig(config=CONFIG.sub_config("paths"))
TRAIN_HYPER_PARAMS = CONFIG.sub_config("train")
IMAGE_PARAMS = CONFIG.sub_config("image")

MARGIN = TRAIN_HYPER_PARAMS["margin"]
IH = IMAGE_PARAMS["H"]
IW = IMAGE_PARAMS["W"]
k_H = IMAGE_PARAMS["k_H"]
k_W = IMAGE_PARAMS["k_W"]
GAMMA = IMAGE_PARAMS["gamma"]

CACHE_LOADER_DIR = PATHS["cache_loader_dir"]

MODEL_PATH = PATHS['model_path']
MODEL_DIR = PATHS['model_dir']
MODEL_META = PATHS['model_meta']

SEED = CONFIG["seed"]
