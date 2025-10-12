"""
日志工具模块

提供分布式训练环境下的日志打印功能
"""

from accelerate import Accelerator


def Logger(msg: str, accelerator: Accelerator = None) -> None:
    """
    打印日志信息（仅主进程输出）

    参数：
        msg: 日志消息内容
        accelerator: Accelerator实例，用于判断是否为主进程
    """
    if  accelerator is not None and accelerator.is_main_process:
        print(msg)
