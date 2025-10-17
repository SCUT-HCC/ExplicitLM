from hydra_zen import instantiate, store, make_config
from .model import ModelConf
from .dataset import DatasetConf
from .logging import LoggingConf
from .training import TrainingConf

# 1. 先生成函数节点
_main_cfg_func = make_config(
    model=ModelConf,
    dataset=DatasetConf,
    logging=LoggingConf,
    training=TrainingConf,
)

# 2. 实例化成纯字典（关键！）
# _main_cfg_dict = instantiate(_main_cfg_func)

# 3. 再注册字典（OmegaConf 就认识了）
store(_main_cfg_func, name="lmconfig", package="_global_")