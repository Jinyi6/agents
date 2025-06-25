from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader

from config import ADMIN_PASSWORD
from core.utils import get_system_status

router = APIRouter()

# 定义一个依赖项，用于从请求头 X-Admin-Password 中获取密码
api_key_header = APIKeyHeader(name="X-Admin-Password", auto_error=False)

def verify_password(password: str = Security(api_key_header)):
    """校验密码"""
    if not password or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing admin password.")
    return True

@router.get(
    "/api/admin/system_status",
    tags=["Admin"],
    summary="查询系统运行状态",
    dependencies=[Security(verify_password)]
)
async def check_system_status():
    """
    [需要密码认证]
    查询服务器的当前状态。

    返回信息包括:
    - 磁盘空间使用情况 (df -h 的效果)
    - 日志目录的空间占用
    """
    try:
        status_info = get_system_status()
        return {
            "message": "System status retrieved successfully.",
            "data": status_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system status: {e}")
