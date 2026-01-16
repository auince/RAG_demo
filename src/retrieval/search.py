import sys
import os
from langchain.retrievers import ContextualCompressionRetriever

# 引入之前的模块
try:
    from src.retrieval.reranker import LocalReranker
    from src.embedding.vector_db import VectorDBManager
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.retrieval.reranker import LocalReranker
    from src.embedding.vector_db import VectorDBManager

class SearchEngine:
    def __init__(self, db_path: str = "./data/vector_store"):
        # 1. 初始化向量库管理器
        self.db_manager = VectorDBManager(persist_dir=db_path)
        
        # 2. 初始化本地 Reranker (连接 vLLM/TEI)
        self.reranker = LocalReranker(
            top_n=5,               # 最终给大模型看前 5 个最相关的块
            score_threshold=0.3    # 过滤掉相关度太低的噪音
        )

    def get_retriever(self):
        """
        返回一个“压缩检索器” (Compression Retriever)。
        流程：VectorDB (Top 20) -> Reranker (Top 5) -> LLM
        """
        # 基础检索器：使用 MMR 获取多样化的 Top 20
        base_retriever = self.db_manager.get_retriever(search_type="mmr", k=20)
        
        # 组合检索器：将 Reranker 挂载到基础检索器之上
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=base_retriever
        )
        
        return compression_retriever