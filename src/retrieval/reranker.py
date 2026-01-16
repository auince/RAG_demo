import requests
import json
from typing import Sequence, List, Optional
from langchain_core.documents import Document
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor

class LocalReranker(BaseDocumentCompressor):
    """
    自定义 Reranker，适配 vLLM 的 Score API。
    文档参考: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#score-api
    """
    # 根据你提供的文档，API 路径通常是 /score 而不是 /v1/score
    endpoint: str = "http://localhost:4062/score"
    model_name: str = "bge-reranker-v2-m3"
    top_n: int = 5
    score_threshold: float = 0.0

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        核心重排序逻辑
        """
        if len(documents) == 0:
            return []

        # 1. 构造请求 Payload (Batch Inference: One-to-Many)
        # text_1: Query (String)
        # text_2: List of Candidates (List[String])
        payload = {
            "model": self.model_name,
            "text_1": query, 
            "text_2": [doc.page_content for doc in documents]
        }

        try:
            # 发送请求
            response = requests.post(
                self.endpoint, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # 调试：如果报错，打印服务端返回的具体信息
            if response.status_code != 200:
                print(f"[Rerank Error] HTTP {response.status_code}: {response.text}")
            
            response.raise_for_status()
            
            # 2. 解析返回结果
            # vLLM 返回格式: {"data": [{"index": 0, "score": 0.9}, ...]}
            results = response.json()
            
            if "data" in results:
                # 确保按 index 排序，因为 API 可能不保证顺序（虽然通常是保序的）
                data_list = results["data"]
                # 创建一个长度正确的 score 列表
                scores = [0.0] * len(documents)
                for item in data_list:
                    idx = item.get("index")
                    score = item.get("score")
                    if idx is not None and idx < len(scores):
                        scores[idx] = score
            else:
                print(f"[Rerank Error] 响应中未找到 'data' 字段: {results}")
                return documents[:self.top_n]

        except Exception as e:
            print(f"[Rerank Warning] 服务调用失败: {e}。返回原始排序。")
            return documents[:self.top_n]

        # 3. 结合分数筛选文档
        final_results = []
        for doc, score in zip(documents, scores):
            if score >= self.score_threshold:
                # 复制 metadata 以避免污染原始对象
                doc_metadata = doc.metadata.copy()
                doc_metadata["relevance_score"] = score
                doc.metadata = doc_metadata
                final_results.append((doc, score))

        # 4. 按分数降序排列
        final_results.sort(key=lambda x: x[1], reverse=True)

        # 5. 返回 Top N
        return [doc for doc, score in final_results[:self.top_n]]

# ==========================================
# 独立执行的测试函数
# ==========================================
def test_reranker_independently():
    print("------- 开始 Rerank 模型连接测试 -------")
    
    query = "DeepSeek是什么?"
    docs = [
        Document(page_content="DeepSeek 是一个强大的开源大语言模型。", metadata={"id": 1}),
        Document(page_content="今天天气真不错。", metadata={"id": 2}),
        Document(page_content="DeepSeek-V3 提供了 128k 的上下文窗口。", metadata={"id": 3}),
    ]
    
    reranker = LocalReranker(top_n=3)
    
    print(f"[1/2] 发送请求到 {reranker.endpoint}...")
    try:
        reranked_docs = reranker.compress_documents(docs, query)
        
        print(f"[2/2] 重排序完成。Top {len(reranked_docs)} 结果:")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get("relevance_score", "N/A")
            print(f"   {i+1}. [Score: {score}] {doc.page_content}...")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_reranker_independently()
