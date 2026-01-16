from typing import List
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document

class HybridSplitter:
    """
    混合分块策略：
    1. 语义分块：基于 Markdown Header (#, ##) 提取章节结构元数据。
    2. 物理分块：基于字符数限制，但尽量不打断代码块。
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 定义需要提取的 Header 层级 (用于生成 Metadata)
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

    def split(self, documents: List[Document]) -> List[Document]:
        print("[Splitter] 开始进行混合分块处理...")
        
        final_chunks = []
        
        # --- 第一步：基于 Header 的语义切分 ---
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False # 保留标题文本在内容中，便于模型理解上下文
        )

        for doc in documents:
            # 1. 提取原始文档的 Metadata (如 source, filename)
            original_metadata = doc.metadata.copy()
            
            # 2. 进行 Header 切分
            # 注意：这会返回一组新的 Document，包含 content 和 header metadata
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            # 3. 合并 Metadata：将 Header 信息与原始 source 信息合并
            for split in header_splits:
                split.metadata.update(original_metadata)

            # --- 第二步：基于字符数的递归切分 ---
            # 定义分隔符，优先级从高到低。
            # 关键："\n```" 放在最前面，防止切断代码块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n```",  # 优先在代码块边界切分
                    "\n\n",   # 其次是段落
                    "\n",     # 再次是行
                    " ",      # 最后是空格
                    ""
                ]
            )
            
            # 对 Header 切分后的结果进行二次切分
            recursive_splits = text_splitter.split_documents(header_splits)
            final_chunks.extend(recursive_splits)

        print(f"[Splitter] 处理完成。原始文档 {len(documents)} -> 分块后 {len(final_chunks)} 个 Chunk。")
        return final_chunks