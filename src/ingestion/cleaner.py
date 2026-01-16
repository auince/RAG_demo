import re
from langchain_core.documents import Document
from typing import List

class DataCleaner:
    """
    负责清洗原始文档内容，移除无用字符和特定噪音。
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗纯文本内容
        """
        # 1. 移除不可见的特殊字符 (保留换行符)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 2. 移除连续的多余空行 (超过3行变2行，保留段落感但压缩空间)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. 移除常见的文档生成器噪音 (根据你的实际文档情况调整)
        # 例如: "Edit on GitHub", "Last updated on..."
        # text = re.sub(r'Edit on GitHub', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    @staticmethod
    def clean_documents(documents: List[Document]) -> List[Document]:
        """
        批量清洗 Document 对象列表
        """
        cleaned_docs = []
        for doc in documents:
            # 清洗 content
            cleaned_content = DataCleaner.clean_text(doc.page_content)
            
            # 如果清洗后内容过短（例如只有标题），则丢弃
            if len(cleaned_content) < 10:
                continue
                
            # 更新 Document 内容，保留 Metadata
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)
            
        return cleaned_docs