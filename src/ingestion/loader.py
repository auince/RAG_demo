import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

class DocLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self) -> List[Document]:
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"目录不存在: {self.data_dir}")

        print(f"[Loader] 正在从 {self.data_dir} 加载文档...")

        # --- 修改开始: 增强 Loader 配置 ---
        # 1. 尝试使用 UTF-8 自动检测
        loader_kwargs = {"autodetect_encoding": True}
        
        try:
            loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs=loader_kwargs,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True # 遇到无法读取的文件跳过，而不是崩溃
            )
            docs = loader.load()
        except Exception as e:
            print(f"⚠️ [Loader] 并发加载出错，尝试单线程回退: {e}")
            # 回退方案
            loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs=loader_kwargs,
                show_progress=True,
                use_multithreading=False
            )
            docs = loader.load()
        # --- 修改结束 ---

        # 过滤掉内容为空的文档
        valid_docs = [doc for doc in docs if doc.page_content and len(doc.page_content.strip()) > 0]
        
        print(f"[Loader] 成功加载 {len(valid_docs)} 个文档文件 (已过滤空文件)。")
        return valid_docs