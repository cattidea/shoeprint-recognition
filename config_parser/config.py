from config_parser.parser import new_config

_CONFIG_PATH = "config.json"
CONFIG = new_config(_CONFIG_PATH)
PATHS = CONFIG.paths
PATHS.init_path()
TRAIN_HYPER_PARAMS = CONFIG.train
IMAGE_PARAMS = CONFIG.image
TEST_PARAMS = CONFIG.test
DOCS = CONFIG.docs

IH = IMAGE_PARAMS["H"]
IW = IMAGE_PARAMS["W"]

CACHE_LOADER_DIR = PATHS["cache_loader_dir"]


MODEL_DIR = PATHS['model_dir']


MARGIN = CONFIG["margin"]
SEED = CONFIG["seed"]
DEBUG = CONFIG["debug"]
