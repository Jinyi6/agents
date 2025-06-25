import time
import logging
import shutil
from pathlib import Path

from config import MAX_RETRIES, log_dir

def retry_step(func):
    """一个装饰器，用于为关键步骤添加重试逻辑"""
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                # 执行函数
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logging.warning(f"[Retry] 步骤 '{func.__name__}' 第 {attempt + 1}/{MAX_RETRIES} 次尝试失败: {e}。将在 2 秒后重试...")
                time.sleep(2)
        logging.error(f"[Failed] 步骤 '{func.__name__}' 在 {MAX_RETRIES} 次尝试后最终失败。")
        raise last_exception
    return wrapper

def get_dir_size(path='.'):
    """计算目录的总大小"""
    total = 0
    path = Path(path)
    if not path.exists():
        return 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def get_system_status():
    """获取系统状态，如磁盘空间和日志目录大小"""
    total, used, free = shutil.disk_usage("/")
    
    # 将字节转换为更易读的格式 (GB)
    total_gb = total / (2**30)
    used_gb = used / (2**30)
    free_gb = free / (2**30)
    
    # 计算日志目录大小 (MB)
    logs_size_mb = get_dir_size(log_dir) / (2**20)

    return {
        "disk_usage": {
            "total_gb": f"{total_gb:.2f}",
            "used_gb": f"{used_gb:.2f}",
            "free_gb": f"{free_gb:.2f}",
            "used_percent": f"{used / total * 100:.2f}%"
        },
        "log_directory_size_mb": f"{logs_size_mb:.2f}",
        "log_directory_path": str(log_dir.resolve())
    }