# 用聊天模型和提示模板构建一个简单的LLM应用程序

在本快速入门中，我们将向你展示如何使用 LangChain 构建一个简单的 LLM 应用。该应用可以将文本从英文翻译成其他语言。这是一个相对简单的 LLM 应用——只需一次 LLM 调用加上一些提示即可。然而，这也是入门 LangChain 的绝佳方式——很多功能只需一些提示和一次 LLM 调用就能实现！

阅读本教程后，你将对以下内容有一个高层次的概览：

*   使用 [语言模型](https://langchain-doc.cn/concepts/chat_models)
*   使用 [提示模板](https://langchain-doc.cn/concepts/prompt_templates)
*   使用 [LangSmith](https://docs.smith.langchain.com/) 调试和跟踪你的应用

让我们开始吧！

## 环境准备

### Jupyter Notebook

本教程和其他教程在 [Jupyter Notebook](https://jupyter.org/) 中运行可能最方便。在交互式环境中浏览指南是更好理解它们的好方法。请参见 [安装指南](https://jupyter.org/install) 获取安装说明。

### 安装

要安装 LangChain，请运行：

```python
# | output: false
# %pip install langchain
# OR
# %conda install langchain -c conda-forge
```

更多细节，请参阅我们的 [安装指南](../how_to/installation.html)。

### LangSmith

许多使用 LangChain 构建的应用会包含多个步骤，并多次调用 LLM。
随着应用变得越来越复杂，能够检查链或代理内部发生了什么变得至关重要。
实现这一点的最佳方式是使用 [LangSmith](https://smith.langchain.com)。

在上述链接注册后，请确保设置环境变量以开始记录追踪：

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="default" # 或其他项目名称
```

或者在 Notebook 中，你可以这样设置：

```python
import getpass
import os
try:
    # 从 .env 文件加载环境变量（需要 `python-dotenv`）
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="输入你的 LangSmith API Key（可选）："
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='输入你的 LangSmith 项目名称（默认 = "default"）：'
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"
```

## 使用语言模型

首先，我们学习如何单独使用语言模型。LangChain 支持许多不同的语言模型，可以互换使用。
有关如何开始使用特定模型的详细信息，请参阅 [支持的集成](https://langchain-doc.cn/integrations/chat/)。

选择模型：

### OpenAI

```shell
pip install -qU "langchain[openai]"
```

```python
import getpass
import os
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")
```

### Anthropic

```shell
pip install -qU "langchain[anthropic]"
```

```python
import getpass
import os
if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")
# Note: Model versions may become outdated. Check https://docs.anthropic.com/en/docs/about-claude/models/overview for latest versions
from langchain.chat_models import init_chat_model
model = init_chat_model("claude-3-7-sonnet-20250219", model_provider="anthropic")
```

[更多模型配置](https://python.langchain.com/docs/tutorials/llm_chain/#using-language-models)

让我们先直接使用模型。[ChatModels](https://langchain-doc.cn/concepts/chat_models) 是 LangChain [Runnables](https://langchain-doc.cn/concepts/runnables/) 的实例，这意味着它们提供了一个与之交互的标准接口。要简单调用模型，我们可以将 [消息](https://langchain-doc.cn/concepts/messages/) 列表传递给 `.invoke` 方法。

```python
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="将以下内容从英文翻译成意大利语"),
    HumanMessage(content="hi!"),
]
model.invoke(messages)
```

注意，ChatModels 接收 [消息](https://langchain-doc.cn/concepts/messages/) 对象作为输入，并生成消息对象作为输出。除了文本内容外，消息对象还包含对话 [角色](https://langchain-doc.cn/concepts/messages/#role) 并保存重要数据，如 [工具调用](https://langchain-doc.cn/concepts/tool_calling/) 和 token 使用统计。

LangChain 还支持通过字符串或 [OpenAI 格式](https://langchain-doc.cn/concepts/messages/#openai-format) 输入聊天模型。以下是等价的：

```python
model.invoke("Hello")
model.invoke([{"role": "user", "content": "Hello"}])
model.invoke([HumanMessage("Hello")])
```

### 流式输出

因为聊天模型是 [Runnables](https://langchain-doc.cn/concepts/runnables/)，它们提供了包括异步和流式调用模式的标准接口。这允许我们从聊天模型中流式获取单个 token：

```python
for token in model.stream(messages):
    print(token.content, end="|")
```

```
|C|iao|!||
```

有关流式输出聊天模型的更多细节，请参阅 [此指南](https://langchain-doc.cn/how_to/chat_streaming/)。

## 提示模板

目前，我们直接将消息列表传入语言模型。这个消息列表从哪里来？通常，它是由用户输入和应用逻辑组合构建的。应用逻辑通常将原始用户输入转换为准备传入语言模型的消息列表。常见转换包括添加系统消息或使用用户输入格式化模板。

[提示模板](https://langchain-doc.cn/concepts/prompt_templates/) 是 LangChain 的概念，用于协助这种转换。它们接受原始用户输入并返回准备传入语言模型的数据（即提示）。

让我们在这里创建一个提示模板，它将接受两个用户变量：

*   `language`：翻译的目标语言
*   `text`：要翻译的文本

```python
from langchain_core.prompts import ChatPromptTemplate
system_template = "将以下内容从英文翻译成 {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
```

注意，`ChatPromptTemplate` 支持在单个模板中使用多个 [消息角色](https://langchain-doc.cn/concepts/messages/#role)。我们将 `language` 参数格式化到系统消息中，将用户 `text` 格式化到用户消息中。

这个提示模板的输入是一个字典。我们可以单独尝试此提示模板，看看它的效果：

```python
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
prompt
```

```
ChatPromptValue(messages=[SystemMessage(content='将以下内容从英文翻译成 Italian', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})])
```

可以看到，它返回一个由两条消息组成的 `ChatPromptValue`。如果我们想直接访问消息列表，可以这样：

```python
prompt.to_messages()
```

```
[SystemMessage(content='将以下内容从英文翻译成 Italian', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]
```

最后，我们可以将格式化后的提示模板传入聊天模型：

```python
response = model.invoke(prompt)
print(response.content)
```

```
Ciao!
```

查看 [LangSmith 跟踪](https://smith.langchain.com/public/3ccc2d5e-2869-467b-95d6-33a577df99a2/r)，可以看到聊天模型收到的确切提示，以及 [token](https://langchain-doc.cn/concepts/tokens/) 使用信息、延迟、[标准模型参数](https://langchain-doc.cn/concepts/chat_models/#standard-parameters)（如 temperature）及其他信息。

## 总结

就是这样！在本教程中，你已经学会了如何创建第一个简单的 LLM 应用。你学会了如何使用语言模型、如何创建提示模板，以及如何通过 LangSmith 获取对应用的优秀可观测性。

这只是成为熟练 AI 工程师所需学习内容的冰山一角。幸运的是——我们还有很多其他资源！

关于 LangChain 核心概念的进一步阅读，请参阅详细的 [概念指南](https://langchain-doc.cn/concepts)。

如果你对这些概念有更具体的问题，请查阅以下 how-to 指南章节：

*   [聊天模型](../how_to/index.html#chat-models)
*   [提示模板](../how_to/index.html#prompt-templates)

以及 LangSmith 文档：

*   [LangSmith](https://docs.smith.langchain.com)