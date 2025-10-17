from hydra_zen import instantiate, store, make_config, builds
from .model import ModelConf
from .dataset import DatasetConf
from .logging import LoggingConf
from .training import TrainingConf

_main_cfg_func = make_config(
    model=ModelConf,
    dataset=DatasetConf,
    logging=LoggingConf,
    training=TrainingConf,
)

store(_main_cfg_func, name="lmconfig", package="_global_")