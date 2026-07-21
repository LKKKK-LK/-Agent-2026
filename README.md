# ⚖️ 法律AI Agent — 智能法律问答系统

> **基于 RAG + 多 Agent 协作的中国法律 AI 助手 | 纯本地运行 · 无需 GPU · 保护隐私**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 项目亮点

### 🤖 多 Agent 协作架构
不是简单的"问答机器人"——系统由 **4 个专属 Agent** 按流程协作：  
**主管Agent（调度）→ 检索Agent（查法条）→ 问答Agent（分析）→ 总结Agent（整理输出）**  
每个 Agent 各司其职，回答质量远超单次 RAG 调用。

### 🔍 增强型法律检索
- **自动提取法律关键词**：口语问题转专业法律术语（"被炒了"→"解除劳动合同 经济补偿"）
- **多维度检索**：原问题 + 多个法律关键词分别检索，合并去重重排
- **相似度过滤**：低于阈值的结果自动丢弃，确保引用准确

### 🏗️ 完整产品级架构
```
浏览器 ↔ FastAPI RESTful API ↔ 多Agent团队 ↔ FAISS向量索引 ↔ Qwen2.5
```
- **API 优先设计**：提供 RESTful 接口，可集成到任何应用
- **多会话管理**：支持同时开多个对话窗口，互不干扰
- **Markdown 渲染**：法律条文、表格、列表等结构化呈现
- **30 部法律全覆盖**：民法典、刑法、劳动法、公司法等

### 🔒 纯本地 · 零成本
- 基于 Ollama + Qwen2.5，所有数据不出本机
- 无需 GPU，纯 CPU 可运行
- 无需任何第三方 API 费用

---

## 📸 预览

```
┌─────────────────────────────────────────────────────┐
│  📋 试用期多久？                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ⚖️ 结论                                            │
│  试用期最长为6个月，具体根据合同期限确定。               │
│                                                     │
│  📜 法律依据                                         │
│  【中华人民共和国劳动法 | 第二十一条】                  │
│  劳动合同可以约定试用期。试用期最长不得超过六个月。     │
│                                                     │
│  💡 分析                                             │
│  - 3个月以上不满1年 → 试用期不超过1个月                │
│  - 1年以上不满3年 → 试用期不超过2个月                  │
│  - 3年以上 → 试用期不超过6个月                        │
│                                                     │
│  ✅ 建议                                             │
│  试用期包含在劳动合同期限内，单独约定无效。             │
│                                                     │
│  📋 试用期多久？             💔 离婚条件               │
│  🛒 过期食品赔偿             🛡️ 正当防卫              │
│  👔 开除员工                                        │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- [Ollama](https://ollama.com)（运行 Qwen2.5）

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载并启动 Qwen2.5
```bash
ollama pull qwen2.5
ollama run qwen2.5
# ⚠️ 保持此终端运行，不要关闭
```

### 3. 启动系统
新开一个终端：
```bash
cd 法律AI-Agent
python api_server.py
```

### 4. 打开使用
浏览器访问：**http://localhost:8000**

---

## 📚 项目结构

```
法律AI-Agent/
├── api_server.py              # FastAPI Web 服务（RESTful 接口）
├── multi_agent_v2.py          # 多 Agent 核心引擎
├── 法律助手.html               # Web 聊天界面（深色/浅色双主题）
├── requirements.txt           # Python 依赖
├── .gitignore
├── README.md
└── LawData/
    ├── vector_index/          # FAISS 向量索引（6330 条法律片段）
    │   ├── faiss_index.bin
    │   └── chunks.json
    └── 原始法律文本/           # 30 部中国法律法规原文（TXT）
```

---

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 大语言模型 | Qwen2.5 (Ollama) | 本地运行，无需联网 |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 | 1024 维中文优化向量 |
| 向量检索 | FAISS (IndexFlatIP) | 6330 条法律文本片段 |
| Web 框架 | FastAPI + Uvicorn | 异步高性能 |
| 前端 | 纯 HTML/CSS/JS + marked.js | 零依赖，开箱即用 |

---

## 📖 API 文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 聊天页面 |
| `/ask?q=问题` | GET | 单次问答 |
| `/chat` | POST | 多轮对话 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger API 文档 |

访问 `http://localhost:8000/docs` 可在线测试所有接口。

---

## 📄 开源许可

本项目基于 MIT 协议开源。
