"""
Taday 金融智能体 — 统一日志配置

所有模块统一使用 logging 输出日志，支持控制台分级输出 + 文件持久化。
在模块中获取日志: from logger import logger; logger.info("msg")
"""
import logging
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "taday.log")

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 创建根 logger
logger = logging.getLogger("taday")
logger.setLevel(logging.DEBUG)

# 控制台 handler (INFO 级别)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_fmt)

# 文件 handler (DEBUG 级别，记录全部)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_fmt)

# 避免重复添加 handler
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
