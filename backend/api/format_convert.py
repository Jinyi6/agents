import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from config import uploads_dir, conversion_tasks
from core.format_convert import convert_paper_format_logic

router = APIRouter()

@router.post("/api/latex_format/convert", tags=["Converter"])
async def create_conversion_task(
    background_tasks: BackgroundTasks,
    content_file: UploadFile = File(..., description="原始论文的 .zip 或 .tar.gz 压缩包"),
    format_file: UploadFile = File(..., description="新格式模板的 .zip 或 .tar.gz 压缩包")
):
    """接收上传文件，创建后台转换任务"""
    run_id = uuid.uuid4().hex
    # 使用 Path 对象确保跨平台兼容性
    content_path = uploads_dir / f"{run_id}_content{''.join(Path(content_file.filename).suffixes)}"
    format_path = uploads_dir / f"{run_id}_format{''.join(Path(format_file.filename).suffixes)}"
    
    # 保存上传的文件
    try:
        with open(content_path, "wb") as buffer:
            shutil.copyfileobj(content_file.file, buffer)
        with open(format_path, "wb") as buffer:
            shutil.copyfileobj(format_file.file, buffer)
    finally:
        content_file.file.close()
        format_file.file.close()
        
    # 初始化任务状态
    conversion_tasks[run_id] = {
        "status": "processing",
        "run_id": run_id,
        "summary": ["INFO: 任务已创建，正在排队等待处理..."],
        "output_path": None
    }
    
    # 将核心逻辑作为后台任务运行
    background_tasks.add_task(convert_paper_format_logic, run_id, str(content_path), str(format_path))
    
    return {"message": "任务已开始处理", "run_id": run_id}

@router.get("/api/latex_format/status/{run_id}", tags=["Converter"])
async def get_conversion_status(run_id: str):
    """根据 run_id 查询任务状态"""
    task = conversion_tasks.get(run_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到指定的任务ID")
    
    response = {
        "run_id": run_id,
        "status": task["status"],
        "summary": task["summary"],
        "download_url": f"/api/latex_format/download/{run_id}" if task["status"] == "completed" else None
    }
    return response

@router.get("/api/latex_format/download/{run_id}", tags=["Converter"])
async def download_converted_file(run_id: str):
    """根据 run_id 下载结果文件"""
    task = conversion_tasks.get(run_id)
    if not task:
        raise HTTPException(status_code=404, detail="找不到指定的任务ID")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未成功完成，无法下载")
        
    output_path = Path(task["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="结果文件未找到，可能已被清理")
        
    return FileResponse(path=output_path, filename=output_path.name, media_type='application/zip')