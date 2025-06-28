import uuid
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from fastapi import APIRouter, BackgroundTasks, HTTPException

from config import style_transfer_tasks
from core.style_transfer_logic import run_style_transfer_logic

router = APIRouter()

# 定义请求体的数据模型
class StyleTransferRequest(BaseModel):
    original_text: str = Field(..., description="需要润色的原始文本")
    must_include_keywords: Optional[List[str]] = Field(default=None, description="必须包含在内的关键词列表")
    reference_keywords: Optional[List[str]] = Field(default=None, description="供参考的关键词列表")
    style_requirements: Optional[List[str]] = Field(default=None, description="风格要求，如 '专业学术', '口语化' 等")
    style_example: Optional[str] = Field(default=None, description="风格参考示例")
    mode: str = Field(default="标准", description="模式: '标准' 或 '专业'")

# 定义响应体的数据模型
class StyleTransferResponse(BaseModel):
    results: List[str]
    suggestions: str

@router.post("/api/style_transfer/run", tags=["Style Transfer"], summary="启动文本润色任务")
async def start_style_transfer_task(
    request: StyleTransferRequest,
    background_tasks_runner: BackgroundTasks
):
    """
    接收文本润色请求，并创建一个后台任务来处理。
    """
    run_id = uuid.uuid4().hex
    
    # 初始化任务状态
    style_transfer_tasks[run_id] = {
        "status": "processing",
        "run_id": run_id,
        "summary": ["INFO: 任务已创建，正在准备执行文本润色..."],
        "result": None
    }
    
    # 将核心逻辑作为后台任务运行
    background_tasks_runner.add_task(run_style_transfer_logic, run_id, request.dict())
    
    return {"message": "文本润色任务已开始处理", "run_id": run_id}

@router.get("/api/style_transfer/status/{run_id}", tags=["Style Transfer"], summary="查询文本润色任务状态")
async def get_style_transfer_status(run_id: str):
    """
    根据 run_id 查询任务的状态。
    """
    task = style_transfer_tasks.get(run_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到指定的任务ID")
    
    response = {
        "run_id": run_id,
        "status": task["status"],
        "summary": task["summary"],
        "result_url": f"/api/style_transfer/results/{run_id}" if task["status"] == "completed" else None
    }
    return response

@router.get("/api/style_transfer/results/{run_id}", response_model=StyleTransferResponse, tags=["Style Transfer"], summary="获取文本润色结果")
async def get_style_transfer_results(run_id: str):
    """
    根据 run_id 获取最终的润色结果和建议。
    """
    task = style_transfer_tasks.get(run_id)
    if not task or task.get("status") != "completed" or not task.get("result"):
        raise HTTPException(status_code=404, detail="任务未完成或结果不存在。")
        
    return task["result"]
