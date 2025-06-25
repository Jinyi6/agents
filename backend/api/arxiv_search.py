import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from config import background_tasks, DEFAULT_KEYWORDS, DEFAULT_MAX_RESULTS_PER_KEYWORD
from core.arxiv_logic import run_arxiv_search_and_process

router = APIRouter()

# 定义请求体的数据模型，用于数据校验
class ArxivSearchRequest(BaseModel):
    keywords: List[str] = Field(default=DEFAULT_KEYWORDS, description="用于搜索的关键词列表")
    start_date: str = Field(..., description="搜索起始日期 (YYYY-MM-DD)", example=date.today().strftime('%Y-%m-%d'))
    end_date: str = Field(..., description="搜索结束日期 (YYYY-MM-DD)", example=date.today().strftime('%Y-%m-%d'))
    max_results: int = Field(default=DEFAULT_MAX_RESULTS_PER_KEYWORD, gt=0, le=500, description="每个关键词获取的最大论文数")
    target_language: Optional[str] = Field(default=None, description="翻译的目标语言，如 'Chinese'。留空则不翻译。")

@router.post("/api/arxiv/start_search", tags=["Arxiv Search"])
async def start_arxiv_search(
    request: ArxivSearchRequest,
    background_tasks_runner: BackgroundTasks
):
    """
    接收参数，创建后台任务来搜索和处理 arXiv 论文。
    """
    run_id = uuid.uuid4().hex
    
    # 初始化任务状态
    background_tasks[run_id] = {
        "status": "processing",
        "run_id": run_id,
        "summary": ["INFO: 任务已创建，正在验证参数并准备开始..."],
        "output_path": None,
        "request_params": request.dict() # 存储请求参数供日志和调试
    }
    
    # 将核心逻辑作为后台任务运行
    background_tasks_runner.add_task(run_arxiv_search_and_process, run_id, request.dict())
    
    return {"message": "ArXiv 搜索任务已开始处理", "run_id": run_id}

@router.get("/api/arxiv/search_status/{run_id}", tags=["Arxiv Search"])
async def get_search_status(run_id: str):
    """根据 run_id 查询 arXiv 搜索任务的状态。"""
    task = background_tasks.get(run_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到指定的任务ID")
    
    response = {
        "run_id": run_id,
        "status": task["status"],
        "summary": task["summary"],
        "download_url": f"/api/arxiv/download_result/{run_id}" if task["status"] == "completed" else None
    }
    return response

@router.get("/api/arxiv/download_result/{run_id}", tags=["Arxiv Search"])
async def download_search_result(run_id: str):
    """根据 run_id 下载 CSV 结果文件。"""
    task = background_tasks.get(run_id)
    if not task or task.get("status") != "completed" or not task.get("output_path"):
        raise HTTPException(status_code=404, detail="任务未完成或结果文件不存在。")
        
    output_path = Path(task["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="结果文件在服务器上未找到，可能已被清理。")
        
    return FileResponse(path=output_path, filename=output_path.name, media_type='text/csv')