"""
====================================================
  多Agent法律问答系统 - 核心引擎
====================================================

架构：
  主管Agent（调度）→ 检索Agent → 问答Agent → 总结Agent → 审核输出

功能：
  - 多轮对话（滑动窗口压缩历史）
  - 多关键词增强检索
  - 结构化法律分析输出
"""

import os
import json
import re
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== 自动获取项目根目录（兼容任何运行路径） =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ===== 配置 =====
INDEX_DIR = os.path.join(PROJECT_ROOT, "LawData", "vector_index")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# 嵌入模型名称（第一次运行会自动下载）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# Ollama 本地模型
OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL = "qwen2.5"

# 滑动窗口配置
MAX_HISTORY_CHARS = 4000

# 检索配置
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.3

# ===== 全局聊天记录 =====
chat_history = []

# ===== 加载模型和数据 =====
print("📦 加载FAISS索引和法律文本片段...")
index = faiss.read_index(FAISS_PATH)
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)
print(f"  索引: {index.ntotal} 向量 | 文本: {len(chunks)} 段\n")

print("🧠 加载BGE嵌入模型...")
bge_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"  加载完成! 维度: {bge_model.get_embedding_dimension()}\n")


# ================================================================
# 底层工具
# ================================================================

def search_law(query: str, top_k: int = None) -> str:
    """从法律数据库中检索相关法律条文"""
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    query_vec = bge_model.encode(query).reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query_vec, max(top_k * 2, 10))

    raw_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunks):
            if score < RETRIEVAL_SCORE_THRESHOLD:
                continue
            chunk = chunks[idx]
            raw_results.append({
                "score": float(score),
                "source": chunk['source'],
                "article": chunk['article'],
                "text": chunk['text'][:400]
            })

    raw_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = raw_results[:top_k]

    if not top_results:
        return "未找到相关法律条文。"

    results = []
    for r in top_results:
        results.append(
            f"【{r['source']} | {r['article']}】(相似度: {r['score']:.2f})\n{r['text']}"
        )
    return "\n\n".join(results)


def search_law_multi(question: str, top_k_per_query: int = 3, max_total: int = 6) -> str:
    """
    多关键词增强检索：
    1. 原问题直接检索
    2. Qwen提取法律关键词分别检索
    3. 合并去重，按分数重排
    """
    # 步骤1：原问题检索
    query_vec = bge_model.encode(question).reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query_vec, top_k_per_query * 2)

    seen = set()
    combined = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunks) and score >= RETRIEVAL_SCORE_THRESHOLD:
            chunk = chunks[idx]
            key = f"{chunk['source']}|{chunk['article']}"
            if key not in seen:
                seen.add(key)
                combined.append({
                    "score": float(score),
                    "source": chunk['source'],
                    "article": chunk['article'],
                    "text": chunk['text'][:400]
                })

    # 步骤2：提取法律关键词
    keyword_prompt = f"""你是一个法律检索专家。请分析用户的问题，提取2-3个最适合搜索法律条文的关键词(或短语)。

要求：
- 每个关键词用尖括号括起来
- 关键词要精确（如"解除劳动合同"而不是"被离职"）
- 如果是多义词，加上限定词

示例：
用户：试用期被辞退有赔偿吗？
输出：<解除劳动合同> <试用期> <经济补偿>

用户：{question}
输出："""

    try:
        keyword_response = ask_qwen(
            "你是一个法律检索关键词提取专家。只输出关键词，不要任何解释。",
            keyword_prompt,
            temperature=0.1
        )
        keywords = re.findall(r'<([^>]+)>', keyword_response)
    except Exception:
        keywords = []

    # 步骤3：每个关键词分别检索
    for kw in keywords[:3]:
        if not kw.strip() or kw.strip() == question:
            continue
        kw_vec = bge_model.encode(kw.strip()).reshape(1, -1).astype(np.float32)
        scores_kw, indices_kw = index.search(kw_vec, top_k_per_query)

        for score, idx in zip(scores_kw[0], indices_kw[0]):
            if idx >= 0 and idx < len(chunks) and score >= RETRIEVAL_SCORE_THRESHOLD:
                chunk = chunks[idx]
                key = f"{chunk['source']}|{chunk['article']}"
                if key not in seen:
                    seen.add(key)
                    combined.append({
                        "score": float(score),
                        "source": chunk['source'],
                        "article": chunk['article'],
                        "text": chunk['text'][:400]
                    })

    # 步骤4：分数重排
    combined.sort(key=lambda x: x["score"], reverse=True)
    top_combined = combined[:max_total]

    if not top_combined:
        return "未找到相关法律条文。"

    results = []
    for r in top_combined:
        results.append(f"【{r['source']} | {r['article']}】\n{r['text']}")
    return "\n\n".join(results)


def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        import re as _re
        clean = ''.join(_re.findall(r'[\d+\-*/.()% ]', expression))
        result = eval(clean)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ================================================================
# 调用Qwen大模型
# ================================================================
def ask_qwen(system_prompt, user_message, temperature=0.1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        },
        timeout=120
    )
    return r.json()["message"]["content"]


# ================================================================
# 滑动窗口：自动压缩历史
# ================================================================
def summarize_history(history):
    """历史过长时，自动总结旧内容，保留最近2轮"""
    if not history:
        return []

    total_chars = sum(len(m["content"]) for m in history)
    if total_chars <= MAX_HISTORY_CHARS:
        return history

    recent = history[-4:] if len(history) >= 4 else history
    old = history[:-4] if len(history) >= 4 else []

    if not old:
        return history

    old_text = ""
    for m in old:
        role = "用户" if m["role"] == "user" else "助手"
        old_text += f"{role}：{m['content'][:200]}\n\n"

    summary_prompt = f"""请将以下对话历史总结成1-2句话，保留关键信息（用户关心的话题、已经讨论过的法律领域）。

需要总结的对话：
{old_text}

总结："""

    summary = ask_qwen(
        "你是一个对话记录员，负责将对话历史简要总结。",
        summary_prompt,
        temperature=0.1
    )

    return [
        {"role": "assistant", "content": f"【之前的对话摘要】{summary.strip()}"}
    ] + recent


def format_history(history):
    """把聊天记录转成文本"""
    if not history:
        return ""
    lines = []
    for m in history:
        role = "用户" if m["role"] == "user" else "助手"
        content = m["content"][:300] + "..." if len(m["content"]) > 300 else m["content"]
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


# ================================================================
# Agent 1：检索Agent
# ================================================================
def agent_retrieval(query):
    system_prompt = """你是一个法律检索专员。你的工作只有一项：从法律数据库中查找相关的法律条文。

【核心规则】
1. 你只能使用 search_law 工具来检索法律条文
2. 如果第一次检索结果不够好，可以换关键词再查一次
3. 你不需要分析或解释法律条文，只需要把原文找出来
4. 返回找到的法律条文即可，不要添加任何额外内容

【搜索技巧】
- 不要把整个问题作为搜索词，要提取核心法律关键词
- 例如用户问"公司不给我交社保怎么办"，应搜"社会保险"而不是整句话
- 如果法律术语有多个表述方式，换词再查（如"解除劳动合同"和"辞退")
- 优先搜索法律名称（如"劳动合同法"）或专业术语

【工具使用格式 - 严格遵循】
<tool>search_law</tool>
<param>query</param>
你的法律搜索词（要精准，不要过长）
<end>

【示例】
好的搜索：<tool>search_law</tool><param>query</param>试用期 解除劳动合同<end>
不好的搜索：<tool>search_law</tool><param>query</param>请问试用期被辞退有没有经济补偿这个法律是怎么规定的<end>

检索到结果后，直接把结果返回，不要添加额外解释。"""
    
    context = f"""用户的问题：{query}

第一步：提取法律关键词，不要直接复制用户口语。
第二步：用提取的关键词通过 search_law 进行检索。
开始。"""
    
    response = ask_qwen(system_prompt, context, temperature=0.1)
    
    tool_match = re.search(r'<tool>(.*?)</tool>', response)
    if tool_match:
        param_match = re.search(r'<param>(.*?)</param>\s*(.*?)\s*<end>', response, re.DOTALL)
        if param_match:
            param_value = param_match.group(2).strip()
            result = search_law_multi(param_value)
            
            if result == "未找到相关法律条文。":
                result = search_law(query, top_k=4)
            
            confirm_prompt = f"""你之前检索了：{param_value}
得到了以下结果：
{result}

请直接返回这些结果原文，不要添加任何分析或解释。"""
            
            return ask_qwen(
                "你是法律检索专员，只返回找到的法律条文原文，不要添加任何内容。",
                confirm_prompt,
                temperature=0.1
            )
    
    return response


# ================================================================
# Agent 2：问答Agent
# ================================================================
def agent_qa(question, legal_texts, history_context=""):
    system_prompt = """你是一位资深法律顾问。你的工作是根据提供的法律条文，分析用户的法律问题。

【分析框架 - 请按以下步骤思考】
步骤1 - 案情识别：明确用户遇到了什么法律问题（劳动争议？合同纠纷？侵权？）
步骤2 - 法律定性：这个问题属于哪个法律领域（劳动法、民法、刑法、商法、行政法等）
步骤3 - 条文匹配：将检索到的法律条文与用户的问题逐条对照
步骤4 - 推理论证：用法律条文的原文来解释用户的情况，给出专业判断
步骤5 - 结论：基于上述分析给出明确结论

【写作规范】
1. 必须注明每条法律依据的来源（哪部法律第几条）
2. 如果涉及金额计算，使用 calculator 工具
3. 区分"法律规定"和"我的理解"，法律条文原文要准确引用
4. 如果法律条文不足以覆盖用户的问题，明确指出来
5. 注意对话历史，如果用户在追问或补充，要结合上下文理解
6. 如果用户的问题描述模糊或信息不足，指出需要补充哪些信息

【引用格式示例】
根据《中华人民共和国劳动合同法》第十九条："劳动合同期限三个月以上不满一年的，试用期不得超过一个月..."

【工具使用格式】
<tool>calculator</tool>
<param>expression</param>
算式
<end>

如果不需要计算，直接给出分析结果。"""
    
    context = f"""用户的问题：{question}

检索到的法律条文：
{legal_texts}

"""
    if history_context:
        context += f"""【对话上下文 - 用户之前聊过的内容】
{history_context}

请注意：历史对话可能包含用户在追问或补充信息，结合上下文理解用户的真实意图。

"""

    context += """请按以下步骤分析：
1. 首先判断这是哪类法律问题
2. 然后将检索到的法律条文与用户问题对照
3. 最后给出专业法律分析

开始分析："""
    
    response = ask_qwen(system_prompt, context, temperature=0.15)
    
    tool_match = re.search(r'<tool>(.*?)</tool>', response)
    if tool_match:
        param_match = re.search(r'<param>(.*?)</param>\s*(.*?)\s*<end>', response, re.DOTALL)
        if param_match and tool_match.group(1).strip() == "calculator":
            param_value = param_match.group(2).strip()
            calc_result = calculator(param_value)
            
            response = ask_qwen(
                system_prompt,
                f"""原问题：{question}
法律条文：
{legal_texts}

计算结果：{calc_result}

请根据这个计算结果和前面的法律条文，完成你的法律分析。引用计算过程。""",
                temperature=0.15
            )
    
    return response


# ================================================================
# Agent 3：总结Agent
# ================================================================
def agent_summarize(question, legal_texts, qa_analysis, history_context=""):
    system_prompt = """你是一位法律文书撰写专家。你的工作是把法律分析结果整理成清晰、易懂、结构化的回答。

【输出结构 - 严格按照以下格式】
===
### 结论
（用1-2句话直接回答用户的核心问题）

### 法律依据
（列出引用的法律条文，注明法律名称和条款编号）

### 详细分析
（展开推理过程，解释法律如何适用于用户的情况，必要时给出计算过程）

### 建议
（基于法律分析，给用户可操作的建议或提醒注意事项）
===

【写作规范】
1. 用用户能听懂的语言表达，避免过于晦涩的法律术语堆砌
2. 保持法律准确性，不能歪曲原意或过度简化
3. 法律条文引用必须标注来源，格式：【法律名称|条款编号】
4. 回答要完整，不要让用户再追问才能理解
5. 如果信息不足以给出确定结论，明确说明存在哪些可能性和影响因素
6. 涉及时间、金额、条件等具体数字要清晰标注"""
    
    context = f"""用户问题：{question}

相关法律条文：
{legal_texts}

法律分析结果：
{qa_analysis}

"""

    if history_context:
        context += f"""【对话上下文】
{history_context}

注意：如果用户在追问，回答时要体现上下文连贯性。

"""

    context += "请整理成一份完整的、用户友好的法律回答。按照：结论 → 法律依据 → 详细分析 → 建议 的结构组织。"
    
    return ask_qwen(system_prompt, context, temperature=0.25)


# ================================================================
# Agent 4：主管Agent
# ================================================================
def agent_supervisor(question, history_context=""):
    """主管Agent - 团队调度员"""
    # 第1步：主管分析问题，拆解任务
    system_prompt = """你是一个项目主管，负责调度一个法律助手团队。
你的团队有三个专家：
1. 检索专员 - 从法律数据库中查找法律条文（搜索关键词要专业、精准）
2. 法律顾问 - 分析法律条文如何适用于具体问题（输出结构化分析）
3. 文书专家 - 将分析结果整理成清晰完整的回答（结论→依据→分析→建议）

【你的工作流程】
1. 分析用户问题：判断属于哪个法律领域（劳动法/民法/刑法/商法等）
2. 拆解子问题：如果用户的问题涉及多个方面，按顺序排列优先级
3. 指派专家：按 检索→分析→总结 的顺序安排工作
4. 审核：检查最终回答是否完整、准确

【注意事项】
- 如果用户在追问之前的话题，要结合对话上下文理解
- 如果问题比较宽泛，先聚焦最核心的部分
- 确保检索结果覆盖用户问题的所有方面
- 如果检索到的法律条文与问题不相关，需要重新检索"""
    
    analysis_input = f"用户问题：{question}\n\n"
    if history_context:
        analysis_input += f"""对话上下文（用户之前聊过的内容）：
{history_context}

注意：结合上下文判断用户是在提问新问题还是追问之前的话题。
"""
    analysis_input += """请分析：
1. 这是什么类型的法律问题？
2. 需要检索哪些法律关键词？
3. 需要注意什么特殊点？

分析结果："""
    
    analysis = ask_qwen(system_prompt, analysis_input, temperature=0.15)
    
    # 第2步：调用检索Agent
    legal_texts = agent_retrieval(question)
    
    # 第3步：调用问答Agent
    qa_result = agent_qa(question, legal_texts, history_context)
    
    # 第4步：调用总结Agent
    final_answer = agent_summarize(question, legal_texts, qa_result, history_context)
    
    # 第5步：主管审核（仅做质量检查，不重新生成内容）
    system_prompt_final = """你是一位法律项目主管。请审核团队的工作成果。

请严格检查：
- 完整性：回答是否覆盖了用户的核心问题？
- 准确性：引用的法律依据是否正确？
- 清晰度：语言是否易懂？

如果满足要求，直接输出一个词：PASS
如果不满足要求，指出具体问题。"""

    final_check = ask_qwen(
        system_prompt_final,
        f"""用户原问题：{question}

团队工作成果：
{final_answer}

请审核以上回答是否满足要求。""",
        temperature=0.1
    )

    # 审核不通过时打印警告，但始终返回 final_answer（避免重复输出）
    if "PASS" not in final_check.strip().upper():
        print(f"    ⚠️ 审核提示：{final_check[:100]}...")
    
    return final_answer


# ================================================================
# 多轮对话主循环
# ================================================================
def conversation_loop():
    global chat_history
    
    print("=" * 70)
    print("  多Agent法律助手团队 (多轮对话版)")
    print("=" * 70)
    print()
    print("  输入 'exit' 退出")
    print("  输入 '!history' 查看聊天记录长度")
    print()
    
    round_num = 0
    
    while True:
        question = input("👤 > ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print(f"\n👋 再见！本次共对话了 {round_num} 轮")
            break
        
        if not question:
            continue
        
        if question == '!history':
            total = sum(len(m["content"]) for m in chat_history)
            print(f"\n  📊 聊天记录：{len(chat_history)} 条消息，共 {total} 字")
            continue
        
        round_num += 1
        chat_history = summarize_history(chat_history)
        history_text = format_history(chat_history)
        
        answer = agent_supervisor(question, history_text)
        
        print(f"\n{'=' * 70}")
        print(f"💡 回答:\n{answer}")
        print(f"{'=' * 70}")
        
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        
        total = sum(len(m["content"]) for m in chat_history)
        print(f"\n  📊 本轮对话已保存 | 累计 {total} 字\n")


# ================================================================
# 入口
# ================================================================
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        answer = agent_supervisor(question, "")
        print(f"\n{'=' * 70}")
        print(f"💡 回答:\n{answer}")
        print(f"{'=' * 70}")
    else:
        conversation_loop()
