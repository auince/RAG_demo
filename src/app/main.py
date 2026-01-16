# 假设这是 src/ingestion/pipeline_demo.py

from src.ingestion.loader import DocLoader
from src.ingestion.cleaner import DataCleaner
from src.ingestion.splitter import HybridSplitter

def run_ingestion(doc_path: str):
    # 1. 加载
    loader = DocLoader(doc_path)
    raw_docs = loader.load()

    # 2. 清洗
    cleaned_docs = DataCleaner.clean_documents(raw_docs)

    # 3. 分块
    splitter = HybridSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split(cleaned_docs)

    # 4. (后续步骤) 传给 Embedding 模块...
    # vector_db.add_documents(chunks)
    
    return chunks

# 测试调用
if __name__ == "__main__":
    chunks = run_ingestion("./data/raw")
    print(f"第一个 Chunk 内容预览:\n{chunks[0].page_content}")
    print(f"第一个 Chunk Metadata:\n{chunks[0].metadata}")