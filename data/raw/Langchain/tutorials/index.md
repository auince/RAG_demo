# 教程

刚接触 LangChain 或 LLM 应用开发？阅读本教程，快速上手并构建您的第一个应用程序。

## 入门

通过构建简单的应用程序来熟悉 LangChain 的开源组件。

如果您想快速上手 [聊天模型](https://langchain-doc.cn/integrations/chat/)、[向量存储](https://langchain-doc.cn/integrations/vectorstores/)或来自特定提供商的其他 LangChain 组件，请查看我们支持的[集成列表](https://langchain-doc.cn/integrations/providers/)。

*   [聊天模型与提示](https://langchain-doc.cn/tutorials/llm_chain)：使用[提示模板](https://langchain-doc.cn/concepts/prompt_templates)和[聊天模型](https://langchain-doc.cn/concepts/chat_models)构建一个简单的 LLM 应用。
*   [语义搜索](https://langchain-doc.cn/tutorials/retrievers)：使用[文档加载器](https://langchain-doc.cn/concepts/document_loaders)、[嵌入模型](https://langchain-doc.cn/concepts/embedding_models/)和[向量存储](https://langchain-doc.cn/concepts/vectorstores/)构建一个基于 PDF 的语义搜索引擎。
*   [分类](https://langchain-doc.cn/tutorials/classification)：使用[聊天模型](https://langchain-doc.cn/concepts/chat_models)和[结构化输出](https://langchain-doc.cn/concepts/structured_outputs/)将文本分类到不同类别或标签中。
*   [抽取](https://langchain-doc.cn/tutorials/extraction)：使用[聊天模型](https://langchain-doc.cn/concepts/chat_models)和[少样本示例](https://langchain-doc.cn/concepts/few_shot_prompting/)从文本及其他非结构化媒体中抽取结构化数据。

更多关于使用所有 LangChain 组件的细节，请参考[操作指南](../how_to.html)。

## 编排

开始使用 [LangGraph](https://langchain-ai.github.io/langgraph/) 将 LangChain 组件组装成功能完善的应用程序。

*   [聊天机器人](https://langchain-doc.cn/tutorials/chatbot)：构建一个带有记忆功能的聊天机器人。
*   [智能体](https://langchain-doc.cn/tutorials/agents)：构建一个可以与外部工具交互的智能体。
*   [检索增强生成（RAG）第一部分](https://langchain-doc.cn/tutorials/rag)：构建一个使用您自己的文档来改进回复的应用。
*   [检索增强生成（RAG）第二部分](https://langchain-doc.cn/tutorials/qa_chat_history)：构建一个具备用户交互记忆和多步检索能力的 RAG 应用。
*   [基于 SQL 的问答](https://langchain-doc.cn/tutorials/sql_qa)：构建一个通过执行 SQL 查询来提供回答的问答系统。
*   [摘要生成](https://langchain-doc.cn/tutorials/summarization)：生成（可能较长）文本的摘要。
*   [基于图数据库的问答](https://langchain-doc.cn/tutorials/graph)：构建一个通过查询图数据库来提供回答的问答系统。

## LangSmith

LangSmith 允许您对 LLM 应用进行细致的追踪、监控和评估。
它可以与 LangChain 无缝集成，您可以在构建过程中用它来检查和调试链条中的每一步。

LangSmith 的文档托管在独立网站上。
您可以在[这里查看 LangSmith 教程](https://docs.smith.langchain.com/)。

### 评估

LangSmith 帮助您评估 LLM 应用的性能。下面的教程是一个很好的入门方式：

*   [评估您的 LLM 应用](https://docs.smith.langchain.com/tutorials/Developers/evaluation)