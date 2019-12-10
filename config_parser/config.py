from config_parser.base import Config, PathConfig

_CONFIG_PATH = "config.json"
CONFIG = Config(_CONFIG_PATH)
PATHS = CONFIG.sub_config("paths", cls=PathConfig)
TRAIN_HYPER_PARAMS = CONFIG.sub_config("train")
IMAGE_PARAMS = CONFIG.sub_config("image")
TEST_PARAMS = CONFIG.sub_config("test")
DOCS = CONFIG.sub_config("docs")

IH = IMAGE_PARAMS["H"]
IW = IMAGE_PARAMS["W"]

CACHE_LOADER_DIR = PATHS["cache_loader_dir"]


MODEL_DIR = PATHS['model_dir']


MARGIN = CONFIG["margin"]
SEED = CONFIG["seed"]
DEBUG = CONFIG["debug"]
