import sys
import os
import requests
from typing import List

# 确保能找到其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class LocalEmbeddings:
    """
    手动封装 vLLM Embedding 接口，避免 LangChain 自动构造复杂 Payload 导致的 400 错误。
    对应端口: 4061
    模型名称: Qwen3-Embedding-8B
    """
    
    def __init__(self):
        self.base_url = "http://localhost:4061/v1/embeddings"
        self.model_name = "Qwen3-Embedding-8B"
        self.api_key = "EMPTY"  # vLLM 本地部署不需要 Key

    @property
    def client(self):
        """
        为了兼容 LangChain 的 Chroma 接口，我们需要返回 self，
        因为 Chroma 会调用 client.embed_documents
        """
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化：手动构造最简单的 HTTP 请求
        """
        # 清洗输入：确保全是字符串，且不为空
        valid_texts = [str(t) for t in texts if t]
        if not valid_texts:
            return []

        payload = {
            "model": self.model_name,
            "input": valid_texts, # 直接传字符串列表，不要传字典
            "encoding_format": "float"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # 提取向量数据，按 index 排序确保顺序一致
            data_points = data["data"]
            data_points.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data_points]
            
        except Exception as e:
            print(f"❌ [Embedding Error] 请求失败: {e}")
            # 打印详细的错误响应以供调试
            if 'response' in locals():
                print(f"   Server Response: {response.text}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """
        单文本向量化
        """
        result = self.embed_documents([text])
        if result:
            return result[0]
        return []

# ==========================================
# 独立执行的测试函数
# ==========================================
def test_embedding_independently():
    print("------- 开始 Embedding 模型连接测试 (手动 HTTP 模式) -------")
    try:
        embedder = LocalEmbeddings()
        test_text = "LangChain是什么"
        print(f"[1/2] 正在发送请求: '{test_text}'...")
        
        vector = embedder.embed_query(test_text)
        
        print(f"[2/2] 成功! 向量维度: {len(vector)}")
        print(f"      前5位: {vector[:5]}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

if __name__ == "__main__":
    test_embedding_independently()
