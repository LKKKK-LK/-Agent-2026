"""
====================================================
  API 服务 - 启动法律AI Agent的Web接口
====================================================

启动方式：python api_server.py
访问方式：浏览器打开 http://localhost:8000
API文档：   http://localhost:8000/docs
"""

import sys
import os
import json
import uuid

# 自动定位项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 导入多Agent系统
print("⏳ 正在加载法律AI Agent系统...")
from multi_agent_v2 import (
    agent_supervisor,
    format_history,
    summarize_history,
    chunks
)

# 多轮对话会话存储
sessions = {}

# 创建FastAPI应用
app = FastAPI(
    title="法律AI Agent - API服务",
    description="基于RAG + 多Agent协作的中国法律智能问答系统",
    version="1.0.0"
)

# 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# 提供Web页面
# ================================================================
@app.get("/")
def serve_web():
    """打开浏览器自动显示法律聊天界面"""
    html_path = os.path.join(PROJECT_ROOT, "法律助手.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "法律助手.html 未找到，请确认文件存在"}


# ================================================================
# 接口1：单次问答
# ================================================================
@app.get("/ask")
def ask_once(q: str = Query(..., description="你的法律问题")):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        answer = agent_supervisor(q.strip(), "")
        return {"code": 0, "question": q.strip(), "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统内部错误: {str(e)}")


# ================================================================
# 接口2：多轮对话
# ================================================================
class ChatRequest(BaseModel):
    question: str
    session_id: str = ""


class ChatResponse(BaseModel):
    code: int
    session_id: str
    question: str
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = req.session_id.strip()
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    try:
        history = summarize_history(history)
        history_text = format_history(history)
        answer = agent_supervisor(req.question.strip(), history_text)

        history.append({"role": "user", "content": req.question.strip()})
        history.append({"role": "assistant", "content": answer})
        sessions[session_id] = history

        return ChatResponse(
            code=0,
            session_id=session_id,
            question=req.question.strip(),
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统内部错误: {str(e)}")


# ================================================================
# 接口3：健康检查
# ================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Qwen2.5 (Ollama)",
        "embedding": "BAAI/bge-large-zh-v1.5",
        "active_sessions": len(sessions)
    }


# ================================================================
# 接口4：获取会话历史
# ================================================================
@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    history = sessions[session_id]
    messages = []
    for m in history:
        messages.append({
            "role": m["role"],
            "content": m["content"][:500]
        })

    return {
        "session_id": session_id,
        "total_messages": len(messages),
        "messages": messages
    }


# ================================================================
# 启动
# ================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  🚀 法律AI Agent API 服务启动")
    print("=" * 60)
    print()
    print("  📖 API文档:     http://localhost:8000/docs")
    print("  🩺 健康检查:     http://localhost:8000/health")
    print("  💬 单次问答:     http://localhost:8000/ask?q=试用期最长多久？")
    print("  🔄 多轮对话:     POST http://localhost:8000/chat")
    print()
    print("  按 Ctrl+C 停止服务")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
