from hydra_zen import instantiate, launch, zen
from config import store,_main_cfg_func          # 触发注册
import hydra

def main(cfg):
    config = instantiate(cfg)
    #输出当前目录
    print(hydra.utils.get_original_cwd())
    # print(hydra.utils.get_cwd())
    print(config)


if __name__ == '__main__':
    # ❌ 旧写法（会去文件系统找 lmconfig.yaml）
    # @hydra.main(version_base="1.2", config_name="lmconfig", config_path="config")

    # ✅ 新写法：用 hydra-zen 的 launch，完全走结构化配置
    launch(_main_cfg_func,main)