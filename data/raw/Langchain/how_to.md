# 操作指南

在这里，你将找到关于 “我该如何…？” 类型问题的答案。  
这些指南是**面向目标**且**具体可操作**的，旨在帮助你完成特定任务。  
若需要概念性解释，请参阅 [概念指南](https://langchain-doc.cn/concepts/)。  
若需要端到端的操作演练，请参阅 [教程](tutorials.html)。  
若需要每个类和函数的全面说明，请参阅 [API 参考](https://python.langchain.com/api_reference/)。

## 安装

*   [如何：安装 LangChain 包](how_to/installation.html)
*   [如何：在不同 Pydantic 版本下使用 LangChain](https://langchain-doc.cn/how_to/pydantic_compatibility)

## 核心功能

此部分重点介绍使用 LangChain 的核心功能。

*   [如何：让模型返回结构化数据](https://langchain-doc.cn/how_to/structured_output)
*   [如何：使用模型调用工具](https://langchain-doc.cn/how_to/tool_calling)
*   [如何：流式运行 Runnables](https://langchain-doc.cn/how_to/streaming)
*   [如何：调试你的 LLM 应用](https://langchain-doc.cn/how_to/debugging)

## 组件

这些是你在构建应用时可以使用的核心模块。

### 聊天模型

[聊天模型](https://langchain-doc.cn/concepts/chat_models) 是新型语言模型，它接收消息作为输入并输出消息。  
查看 [支持的集成](https://langchain-doc.cn/integrations/chat/) 以了解如何使用特定提供商的聊天模型。

*   [如何：一行初始化任意模型](https://langchain-doc.cn/how_to/chat_models_universal_init/)
*   [如何：使用本地模型](https://langchain-doc.cn/how_to/local_llms)
*   [如何：进行函数/工具调用](https://langchain-doc.cn/how_to/tool_calling)
*   [如何：让模型返回结构化输出](https://langchain-doc.cn/how_to/structured_output)
*   [如何：缓存模型响应](https://langchain-doc.cn/how_to/chat_model_caching)
*   [如何：获取日志概率](https://langchain-doc.cn/how_to/logprobs)
*   [如何：创建自定义聊天模型类](https://langchain-doc.cn/how_to/custom_chat_model)
*   [如何：流式返回响应](https://langchain-doc.cn/how_to/chat_streaming)
*   [如何：跟踪令牌使用情况](https://langchain-doc.cn/how_to/chat_token_usage_tracking)
*   [如何：跨提供商跟踪响应元数据](https://langchain-doc.cn/how_to/response_metadata)
*   [如何：使用聊天模型调用工具](https://langchain-doc.cn/how_to/tool_calling)
*   [如何：流式调用工具](https://langchain-doc.cn/how_to/tool_streaming)
*   [如何：处理速率限制](https://langchain-doc.cn/how_to/chat_model_rate_limiting)
*   [如何：让工具按少量示例工作](https://langchain-doc.cn/how_to/tools_few_shot)
*   [如何：绑定模型特定格式工具](https://langchain-doc.cn/how_to/tools_model_specific)
*   [如何：强制调用特定工具](https://langchain-doc.cn/how_to/tool_choice)
*   [如何：直接向模型传入多模态数据](https://langchain-doc.cn/how_to/multimodal_inputs/)

### 消息

[消息](https://langchain-doc.cn/concepts/messages) 是聊天模型的输入和输出。它们包含 `content`（内容）和 `role`（角色），用于描述消息来源。

*   [如何：裁剪消息](https://langchain-doc.cn/how_to/trim_messages/)
*   [如何：过滤消息](https://langchain-doc.cn/how_to/filter_messages/)
*   [如何：合并同类连续消息](https://langchain-doc.cn/how_to/merge_message_runs/)

### 提示模板

[提示模板](https://langchain-doc.cn/concepts/prompt_templates) 负责将用户输入格式化为可传递给语言模型的形式。

*   [如何：使用少量示例](https://langchain-doc.cn/how_to/few_shot_examples)
*   [如何：在聊天模型中使用少量示例](https://langchain-doc.cn/how_to/few_shot_examples_chat/)
*   [如何：部分格式化提示模板](https://langchain-doc.cn/how_to/prompts_partial)
*   [如何：组合提示模板](https://langchain-doc.cn/how_to/prompts_composition)
*   [如何：使用多模态提示](https://langchain-doc.cn/how_to/multimodal_prompts/)

### 示例选择器

[示例选择器](https://langchain-doc.cn/concepts/example_selectors) 用于选择适当的少量示例传递给提示模板。

*   [如何：使用示例选择器](https://langchain-doc.cn/how_to/example_selectors)
*   [如何：按长度选择示例](https://langchain-doc.cn/how_to/example_selectors_length_based)
*   [如何：按语义相似度选择示例](https://langchain-doc.cn/how_to/example_selectors_similarity)
*   [如何：按语义 ngram 重叠选择示例](https://langchain-doc.cn/how_to/example_selectors_ngram)
*   [如何：按最大边际相关选择示例](https://langchain-doc.cn/how_to/example_selectors_mmr)
*   [如何：从 LangSmith 少量示例数据集中选择示例](https://langchain-doc.cn/how_to/example_selectors_langsmith/)

### LLM

LangChain 所称的 [LLMs](https://langchain-doc.cn/concepts/text_llms) 是早期语言模型形式，它接收字符串输入并输出字符串。

*   [如何：缓存模型响应](https://langchain-doc.cn/how_to/llm_caching)
*   [如何：创建自定义 LLM 类](https://langchain-doc.cn/how_to/custom_llm)
*   [如何：流式返回响应](https://langchain-doc.cn/how_to/streaming_llm)
*   [如何：跟踪令牌使用情况](https://langchain-doc.cn/how_to/llm_token_usage_tracking)
*   [如何：使用本地模型](https://langchain-doc.cn/how_to/local_llms)

### 输出解析器

[输出解析器](https://langchain-doc.cn/concepts/output_parsers) 负责将 LLM 输出解析为更结构化的格式。

*   [如何：从消息对象解析文本](https://langchain-doc.cn/how_to/output_parser_string)
*   [如何：使用输出解析器将 LLM 响应解析为结构化数据](https://langchain-doc.cn/how_to/output_parser_structured)
*   [如何：解析 JSON 输出](https://langchain-doc.cn/how_to/output_parser_json)
*   [如何：解析 XML 输出](https://langchain-doc.cn/how_to/output_parser_xml)
*   [如何：解析 YAML 输出](https://langchain-doc.cn/how_to/output_parser_yaml)
*   [如何：在解析错误时重试](https://langchain-doc.cn/how_to/output_parser_retry)
*   [如何：尝试修复输出解析错误](https://langchain-doc.cn/how_to/output_parser_fixing)
*   [如何：编写自定义输出解析器类](https://langchain-doc.cn/how_to/output_parser_custom)

### 文档加载器

[文档加载器](https://langchain-doc.cn/concepts/document_loaders) 负责从各种来源加载文档。

*   [如何：加载 PDF 文件](https://langchain-doc.cn/how_to/document_loader_pdf)
*   [如何：加载网页](https://langchain-doc.cn/how_to/document_loader_web)
*   [如何：加载 CSV 数据](https://langchain-doc.cn/how_to/document_loader_csv)
*   [如何：从目录加载数据](https://langchain-doc.cn/how_to/document_loader_directory)
*   [如何：加载 HTML 数据](https://langchain-doc.cn/how_to/document_loader_html)
*   [如何：加载 JSON 数据](https://langchain-doc.cn/how_to/document_loader_json)
*   [如何：加载 Markdown 数据](https://langchain-doc.cn/how_to/document_loader_markdown)
*   [如何：加载 Microsoft Office 数据](https://langchain-doc.cn/how_to/document_loader_office_file)
*   [如何：编写自定义文档加载器](https://langchain-doc.cn/how_to/document_loader_custom)

### 文本拆分器

[文本拆分器](https://langchain-doc.cn/concepts/text_splitters) 将文档拆分为可用于检索的块。

*   [如何：递归拆分文本](https://langchain-doc.cn/how_to/recursive_text_splitter)
*   [如何：拆分 HTML](https://langchain-doc.cn/how_to/split_html)
*   [如何：按字符拆分](https://langchain-doc.cn/how_to/character_text_splitter)
*   [如何：拆分代码](https://langchain-doc.cn/how_to/code_splitter)
*   [如何：按 Markdown 标题拆分](https://langchain-doc.cn/how_to/markdown_header_metadata_splitter)
*   [如何：递归拆分 JSON](https://langchain-doc.cn/how_to/recursive_json_splitter)
*   [如何：按语义拆分文本](https://langchain-doc.cn/how_to/semantic-chunker)
*   [如何：按令牌拆分](https://langchain-doc.cn/how_to/split_by_token)

### 嵌入模型

[嵌入模型](https://langchain-doc.cn/concepts/embedding_models) 将文本转换为数值表示。  
查看 [支持的集成](https://langchain-doc.cn/integrations/text_embedding/) 以了解如何使用特定提供商的嵌入模型。

*   [如何：嵌入文本数据](https://langchain-doc.cn/how_to/embed_text)
*   [如何：缓存嵌入结果](https://langchain-doc.cn/how_to/caching_embeddings)
*   [如何：创建自定义嵌入类](https://langchain-doc.cn/how_to/custom_embeddings)

### 向量数据库

[向量数据库](https://langchain-doc.cn/concepts/vectorstores) 是可以高效存储和检索嵌入的数据库。  
查看 [支持的集成](https://langchain-doc.cn/integrations/vectorstores/) 以了解如何使用特定提供商的向量数据库。

*   [如何：使用向量数据库进行数据检索](https://langchain-doc.cn/how_to/vectorstores)

### 检索器

[检索器](https://langchain-doc.cn/concepts/retrievers) 用于接收查询并返回相关文档。

*   [如何：使用向量数据库进行检索](https://langchain-doc.cn/how_to/vectorstore_retriever)
*   [如何：生成多条查询以进行数据检索](https://langchain-doc.cn/how_to/MultiQueryRetriever)
*   [如何：使用上下文压缩压缩检索到的数据](https://langchain-doc.cn/how_to/contextual_compression)
*   [如何：编写自定义检索器类](https://langchain-doc.cn/how_to/custom_retriever)
*   [如何：为检索结果添加相似度分数](https://langchain-doc.cn/how_to/add_scores_retriever)
*   [如何：组合多个检索器的结果](https://langchain-doc.cn/how_to/ensemble_retriever)
*   [如何：重新排序检索结果以减少“中间丢失”问题](https://langchain-doc.cn/how_to/long_context_reorder)
*   [如何：为每个文档生成多个嵌入](https://langchain-doc.cn/how_to/multi_vector)
*   [如何：检索文档的完整块](https://langchain-doc.cn/how_to/parent_document_retriever)
*   [如何：生成元数据过滤器](https://langchain-doc.cn/how_to/self_query)
*   [如何：创建时间加权检索器](https://langchain-doc.cn/how_to/time_weighted_vectorstore)
*   [如何：使用混合向量和关键词检索](https://langchain-doc.cn/how_to/hybrid)

### 索引

索引用于保持向量数据库与底层数据源同步。

*   [如何：重新索引数据以保持向量数据库与数据源同步](https://langchain-doc.cn/how_to/indexing)

### 工具

LangChain [工具](https://langchain-doc.cn/concepts/tools) 包含工具描述（供模型调用）以及工具实现函数。  
完整工具列表请参见 [这里](https://langchain-doc.cn/integrations/tools/)。

*   [如何：创建工具](https://langchain-doc.cn/how_to/custom_tools)
*   [如何：使用内置工具和工具包](https://langchain-doc.cn/how_to/tools_builtin)
*   [如何：使用聊天模型调用工具](https://langchain-doc.cn/how_to/tool_calling)
*   [如何：将工具输出传递给聊天模型](https://langchain-doc.cn/how_to/tool_results_pass_to_model)
*   [如何：向工具传递运行时值](https://langchain-doc.cn/how_to/tool_runtime)
*   [如何：为工具添加人工干预环节](https://langchain-doc.cn/how_to/tools_human)
*   [如何：处理工具错误](https://langchain-doc.cn/how_to/tools_error)
*   [如何：强制模型调用特定工具](https://langchain-doc.cn/how_to/tool_choice)
*   [如何：禁用并行工具调用](https://langchain-doc.cn/how_to/tool_calling_parallel)
*   [如何：访问工具的 `RunnableConfig`](https://langchain-doc.cn/how_to/tool_configure)
*   [如何：流式获取工具事件](https://langchain-doc.cn/how_to/tool_stream_events)
*   [如何：从工具返回产物](https://langchain-doc.cn/how_to/tool_artifacts/)
*   [如何：将 Runnables 转换为工具](https://langchain-doc.cn/how_to/convert_runnable_to_tool)
*   [如何：为模型添加临时工具调用能力](https://langchain-doc.cn/how_to/tools_prompting)
*   [如何：向工具传递运行时密钥](https://langchain-doc.cn/how_to/runnable_runtime_secrets)

### 多模态

*   [如何：直接向模型传入多模态数据](https://langchain-doc.cn/how_to/multimodal_inputs/)
*   [如何：使用多模态提示](https://langchain-doc.cn/how_to/multimodal_prompts/)

### 代理（Agents）

*   [如何：使用传统 LangChain 代理（AgentExecutor）](https://langchain-doc.cn/how_to/agent_executor)
*   [如何：从传统 LangChain 代理迁移到 LangGraph](https://langchain-doc.cn/how_to/migrate_agent)

### 回调（Callbacks）

[回调](https://langchain-doc.cn/concepts/callbacks) 可让你在 LLM 应用的不同阶段挂钩操作。

*   [如何：在运行时传入回调](https://langchain-doc.cn/how_to/callbacks_runtime)
*   [如何：将回调附加到模块](https://langchain-doc.cn/how_to/callbacks_attach)
*   [如何：在模块构造函数中传入回调](https://langchain-doc.cn/how_to/callbacks_constructor)
*   [如何：创建自定义回调处理器](https://langchain-doc.cn/how_to/custom_callbacks)
*   [如何：在异步环境中使用回调](https://langchain-doc.cn/how_to/callbacks_async)
*   [如何：分发自定义回调事件](https://langchain-doc.cn/how_to/callbacks_custom_events)

### 自定义

LangChain 的所有组件都可以轻松扩展以支持自定义版本。

*   [如何：创建自定义聊天模型类](https://langchain-doc.cn/how_to/custom_chat_model)
*   [如何：创建自定义 LLM 类](https://langchain-doc.cn/how_to/custom_llm)
*   [如何：创建自定义嵌入类](https://langchain-doc.cn/how_to/custom_embeddings)
*   [如何：编写自定义检索器类](https://langchain-doc.cn/how_to/custom_retriever)
*   [如何：编写自定义文档加载器](https://langchain-doc.cn/how_to/document_loader_custom)
*   [如何：编写自定义输出解析器类](https://langchain-doc.cn/how_to/output_parser_custom)
*   [如何：创建自定义回调处理器](https://langchain-doc.cn/how_to/custom_callbacks)
*   [如何：定义自定义工具](https://langchain-doc.cn/how_to/custom_tools)
*   [如何：分发自定义回调事件](https://langchain-doc.cn/how_to/callbacks_custom_events)

### 序列化

*   [如何：保存和加载 LangChain 对象](https://langchain-doc.cn/how_to/serialization)