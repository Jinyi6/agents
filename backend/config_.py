import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

# --- 用户需要配置的参数 ---
# 建议使用环境变量或安全的密钥管理服务
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
OPENAI_API_BASE = os.getenv("apibase", "[https://api.openai.com/v1](https://api.openai.com/v1)")
MODEL_NAME = os.getenv("qwen3", "qwen3-4b")

# 全局重试次数
MAX_RETRIES = 3

# 管理接口的密码
ADMIN_PASSWORD = "jinyi6"

# --- arXiv 搜索和翻译的默认配置 ---
# 这些值可以在 API 请求中被覆盖
DEFAULT_KEYWORDS = ["large language model RL", "LLM RFT", "LLM Reinforcement Learning Finetuning"]
DEFAULT_MAX_RESULTS_PER_KEYWORD = 10
TRANSLATION_BATCH_SIZE = 5 # 一次并发翻译的摘要数量
MAX_CONCURRENT_TRANSLATIONS = 3 # 最大并发 OpenAI API 调用数



# --- 目录设置 ---
# 创建主目录和子目录
log_dir = Path("./latex_format_logs")
uploads_dir = log_dir / "uploads"
outputs_dir = log_dir / "outputs"
workspace_dir = log_dir / "workspace"


# --- 日志配置 ---
def setup_logging():
    """配置日志记录器"""
    log_dir.mkdir(exist_ok=True)
    uploads_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    workspace_dir.mkdir(exist_ok=True)
    
    log_file_name = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file_path = log_dir / log_file_name
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file_path, 'a', 'utf-8'), logging.StreamHandler(sys.stdout)],
        force=True
    )
    logging.info("Logging configured.")
    if "xxxxxxxx" in OPENAI_API_KEY:
        logging.warning("警告: 您似乎正在使用一个占位符API密钥。请将 'OPENAI_API_KEY' 设置为您的真实密钥。")

# --- 全局变量 ---
# 模拟数据库来存储每个任务的状态
conversion_tasks: Dict[str, Dict] = {}
background_tasks: Dict[str, Dict] = {}
style_transfer_tasks: Dict[str, Dict] = {} # 新增：用于存储文本润色任务的状态

