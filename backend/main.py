import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import arxiv_search, format_convert, admin, style_transfer
from config import setup_logging

# --- 初始化 ---
# 配置日志
setup_logging()

# 创建 FastAPI 应用实例
app = FastAPI(
    title="LaTeX Format Converter API",
    description="Do It by Agent",
    version="2.3.0"
)

# --- 中间件配置 ---
# 配置 CORS 跨域资源共享
origins = [
    "https://agentai.top",
    "https://www.agentai.top",
    "https://agents-frontend.onrender.com",
    "http://localhost", # 本地开发测试
    "http://127.0.0.1" # 本地开发测试
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 路由配置 ---
# 包含来自其他文件的 API 路由
app.include_router(arxiv_search.router)
app.include_router(format_convert.router)
app.include_router(admin.router)
app.include_router(style_transfer.router) 


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the DIA Agent API. Visit /docs for API documentation."}


# --- 启动器 ---
if __name__ == "__main__":
    print("="*60)
    print("启动 DIA Agent Web 服务 (版本 2.1)...")
    print("访问 http://127.0.0.1:8000/docs 查看 API 文档和进行测试。")
    print("="*60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)