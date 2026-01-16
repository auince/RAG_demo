# 文件路径: src/app/chain.py

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 确保能找到其他模块 (适配相对导入问题)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompts import get_rag_prompt
from src.retrieval.search import SearchEngine

def format_docs(docs):
    """
    将检索到的 Documents 列表格式化为 XML 字符串，供 Prompt 使用。
    同时保留文件名，方便溯源。
    """
    formatted = []
    for i, doc in enumerate(docs):
        # 获取 metadata，如果没有 source 字段则标记未知
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content
        formatted.append(f"<doc id='{i}' source='{source}'>\n{content}\n</doc>")
    return "\n\n".join(formatted)

def build_rag_chain():
    """
    构建 LCEL (LangChain Expression Language) 执行链
    """
    # 1. 准备组件
    try:
        llm = DeepSeekClient().get_llm()
        retriever = SearchEngine().get_retriever()
        prompt = get_rag_prompt()
    except Exception as e:
        raise RuntimeError(f"初始化 RAG 组件失败: {e}")

    # 2. 定义链结构
    # input 字典 -> retriever 找文档 -> format_docs 格式化 -> prompt -> llm -> 字符串输出
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain