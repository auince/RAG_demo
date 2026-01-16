import gradio as gr
import sys
import os
import time

# --- 路径修正 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# --- 导入业务逻辑 ---
from src.app.chain import build_rag_chain
from src.ingestion.loader import DocLoader
from src.ingestion.cleaner import DataCleaner
from src.ingestion.splitter import HybridSplitter
from src.embedding.vector_db import VectorDBManager

# ==========================================
# 逻辑函数定义
# ==========================================

def rebuild_index_logic(doc_path):
    if not doc_path or not os.path.exists(doc_path):
        return "❌ 错误：路径不存在，请检查输入。"
    
    try:
        yield "📂 [1/4] 正在扫描加载文档..."
        loader = DocLoader(doc_path)
        raw_docs = loader.load()
        
        yield f"🧹 [2/4] 加载成功 ({len(raw_docs)}个文件)，正在清洗..."
        cleaned_docs = DataCleaner.clean_documents(raw_docs)
        
        yield "✂️ [3/4] 正在进行智能分块..."
        splitter = HybridSplitter()
        chunks = splitter.split(cleaned_docs)
        
        yield f"🧠 [4/4] 正在向量化 {len(chunks)} 个片段..."
        db_manager = VectorDBManager()
        db_manager.create_index(chunks, force_rebuild=True)
        
        yield f"✅ 成功！索引重建完成。\n共处理 {len(chunks)} 个片段。"
        
    except Exception as e:
        yield f"❌ 错误: {str(e)}"

def chat_response_logic(message, history):
    if not message:
        return
    try:
        chain = build_rag_chain()
        partial_response = ""
        for chunk in chain.stream(message):
            partial_response += chunk
            yield partial_response
    except Exception as e:
        yield f"⚠️ 发生错误: {str(e)}"

# ==========================================
# UI 布局构建
# ==========================================

with gr.Blocks(title="DeepSeek DevDocs RAG", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 📚 DeepSeek 开发文档助手 (RAG)")
    
    with gr.Row():
        # --- 左侧：设置区 ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ 知识库管理")
            path_input = gr.Textbox(
                label="文档目录路径",
                value="./data/raw", 
                placeholder="/path/to/docs"
            )
            rebuild_btn = gr.Button("🔄 重建索引", variant="primary")
            status_output = gr.Textbox(label="系统状态", value="就绪", interactive=False, lines=4)

        # --- 右侧：聊天区 ---
        with gr.Column(scale=4):
            # 关键修改 1: 显式指定 type="messages"
            chatbot = gr.Chatbot(
                height=700,
                avatar_images=(None, "https://img.icons8.com/color/48/bot.png"),
                label="对话历史"
            )
            
            msg = gr.Textbox(label="输入你的问题", lines=2, autofocus=True)
            
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], value="🗑️ 清除历史")
                submit_btn = gr.Button("🚀 发送", variant="primary")

    # ==========================================
    # 事件绑定 (关键修改部分)
    # ==========================================
    
    rebuild_btn.click(rebuild_index_logic, inputs=[path_input], outputs=[status_output])

    # 关键修改 2: 适配字典格式的 user_turn
    def user_turn(user_message, history):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    # 关键修改 3: 适配字典格式的 bot_turn
    def bot_turn(history):
        # 1. 获取最后一条用户消息
        user_msg_data = history[-1]["content"] 
        
        # --- 关键修复：清洗 Gradio 的多模态数据格式 ---
        user_message = ""
        if isinstance(user_msg_data, str):
            user_message = user_msg_data
        elif isinstance(user_msg_data, list):
            # Gradio 新版可能返回 [{'text': '...', 'type': 'text'}]
            for item in user_msg_data:
                if isinstance(item, dict) and item.get("type") == "text":
                    user_message = item.get("text", "")
                    break
            # 如果没找到 text 字段，兜底转字符串
            if not user_message:
                user_message = str(user_msg_data)
        else:
            user_message = str(user_msg_data)
        # -------------------------------------------

        # 2. 追加一个空的 Assistant 消息占位
        history.append({"role": "assistant", "content": ""})
        
        # 3. 调用 RAG 逻辑 (确保传入的是纯字符串)
        generator = chat_response_logic(user_message, history[:-1])
        
        for chunk in generator:
            history[-1]["content"] = chunk
            yield history

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, chatbot, chatbot
    )
    submit_btn.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, chatbot, chatbot
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)