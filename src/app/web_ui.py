import streamlit as st
import sys
import os

# 将项目根目录加入路径，防止找不到模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.app.chain import build_rag_chain
from src.ingestion.loader import DocLoader
from src.ingestion.cleaner import DataCleaner
from src.ingestion.splitter import HybridSplitter
from src.embedding.vector_db import VectorDBManager

# 页面配置
st.set_page_config(page_title="DevDocs RAG", layout="wide")
st.title("📚 DeepSeek 开发文档助手")

# --- 侧边栏：数据管理 ---
with st.sidebar:
    st.header("知识库管理")
    doc_path = st.text_input("文档目录路径", value="./data/raw")
    
    if st.button("🔄 重建索引 (Rebuild Index)"):
        with st.status("正在处理数据...", expanded=True) as status:
            try:
                # 1. 加载
                st.write("📂 加载文档...")
                loader = DocLoader(doc_path)
                raw_docs = loader.load()
                
                # 2. 清洗
                st.write("🧹 清洗数据...")
                cleaned_docs = DataCleaner.clean_documents(raw_docs)
                
                # 3. 分块
                st.write("✂️ 智能分块...")
                splitter = HybridSplitter()
                chunks = splitter.split(cleaned_docs)
                
                # 4. 向量化
                st.write("🧠 向量化并存储 (这可能需要一会)...")
                db_manager = VectorDBManager()
                db_manager.create_index(chunks, force_rebuild=True)
                
                status.update(label="✅ 索引构建完成!", state="complete", expanded=False)
                st.success(f"成功处理 {len(chunks)} 个片段。")
            except Exception as e:
                st.error(f"出错: {str(e)}")

# --- 主界面：聊天 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("如何使用这个框架的 API?"):
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 初始化链 (建议结合 st.cache_resource 优化加载速度)
            chain = build_rag_chain()
            
            # 流式输出
            chunks = chain.stream(prompt)
            for chunk in chunks:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"生成回答时出错: {e}")
            full_response = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": full_response})