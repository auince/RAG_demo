import os
import shutil
import sys
from typing import List, Optional

# 引入 LangChain 的核心库
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# 引入我们刚才写的本地 Embedding 封装
# 这里的 import 路径是为了兼顾“作为模块运行”和“直接运行脚本”
try:
    from src.embedding.embedder import LocalEmbeddings
except ImportError:
    # 如果直接运行此文件用于测试，需要调整路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.embedding.embedder import LocalEmbeddings

class VectorDBManager:
    """
    向量数据库管理器 (基于 ChromaDB)。
    负责数据的持久化存储、加载和基础检索。
    """
    
    def __init__(self, persist_dir: str = "./data/vector_store"):
        self.persist_dir = persist_dir
        self.embedding_fn = LocalEmbeddings() # 实例化本地模型连接器
        
        # 确保目录存在
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)

    def create_index(self, chunks: List[Document], force_rebuild: bool = False) -> Chroma:
        """
        从文档块构建新的向量索引。
        :param force_rebuild: 如果为 True，会删除旧的数据库文件重新构建。
        """
        if force_rebuild and os.path.exists(self.persist_dir):
            print(f"[VectorDB] 正在清理旧数据: {self.persist_dir}")
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)

        print(f"[VectorDB] 开始构建索引，共 {len(chunks)} 个片段...")
        
        # Chroma.from_documents 会自动处理 Embedding 并持久化到磁盘
        # collection_name 类似于关系型数据库的 Table Name
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn.client, # 传入 LangChain 兼容的 client
            persist_directory=self.persist_dir,
            collection_name="dev_docs_collection"
        )
        
        print(f"[VectorDB] 索引构建完成并保存至: {self.persist_dir}")
        return vector_store

    def load_index(self) -> Chroma:
        """
        加载已存在的向量数据库。
        """
        if not os.path.exists(self.persist_dir) or not os.listdir(self.persist_dir):
            raise FileNotFoundError(f"向量库不存在或为空: {self.persist_dir}，请先运行构建流程。")

        print(f"[VectorDB] 正在加载现有索引: {self.persist_dir}")
        vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn.client,
            collection_name="dev_docs_collection"
        )
        return vector_store

    def get_retriever(self, search_type="mmr", k=5) -> VectorStoreRetriever:
        """
        获取检索器接口 (供 Chain 使用)
        :param search_type: 'similarity' (默认) 或 'mmr' (多样性排序)
        """
        vector_store = self.load_index()
        
        # 配置检索参数
        # MMR (Maximal Marginal Relevance) 能有效防止检索到内容几乎一样的重复片段
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k, 
                "fetch_k": k * 4  # MMR 算法先取 20 个再筛选出 5 个
            }
        )
        return retriever

# ==========================================
# 独立执行的测试函数
# ==========================================
def test_vector_db():
    print("------- 开始 VectorDB 读写测试 -------")
    
    # 1. 准备测试数据
    test_docs = [
        Document(page_content="DeepSeek API 的 Context Window 是 128k。", metadata={"source": "doc_v1.md"}),
        Document(page_content="使用 Python 调用 DeepSeek 需要安装 openai 库。", metadata={"source": "doc_v2.md"}),
        Document(page_content="苹果是一种水果，富含维生素。", metadata={"source": "irrelevant.md"}), 
    ]
    
    db_path = "./data/test_vector_store"
    manager = VectorDBManager(persist_dir=db_path)
    
    try:
        # 2. 构建索引 (强制重建)
        print("[1/3] 写入测试数据...")
        manager.create_index(test_docs, force_rebuild=True)
        
        # 3. 加载索引并搜索
        print("[2/3] 加载并执行相似度搜索...")
        vector_store = manager.load_index()
        
        # 搜索关于 API 的内容
        query = "DeepSeek 上下文窗口多大？"
        results = vector_store.similarity_search(query, k=1)
        
        print(f"[3/3] 搜索结果 (Query: {query}):")
        if results:
            top_doc = results[0]
            print(f"   --> 内容: {top_doc.page_content}")
            print(f"   --> 来源: {top_doc.metadata}")
            
            if "128k" in top_doc.page_content:
                print("✅ 检索成功：准确匹配到核心信息。")
            else:
                print("⚠️ 检索结果不准确，请检查 Embedding 模型是否正常。")
        else:
            print("❌ 未找到任何结果。")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        # 这里的 import 错误通常是因为没运行 embedder 的服务
        if "Connection" in str(e):
            print("提示: 请确保 vLLM Embedding 服务已在 4061 端口启动。")

    # 清理测试产生的垃圾文件
    # shutil.rmtree(db_path) 

if __name__ == "__main__":
    test_vector_db()