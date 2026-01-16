# 构建一个聊天机器人

> 注
> 本教程之前使用了 [RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html) 抽象。你仍然可以在 [v0.2 文档](https://python.langchain.com/v0.2/docs/tutorials/chatbot/) 中访问该版本的内容。
> 
> 自 LangChain v0.3 起，我们推荐用户使用 [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) 来在新的 LangChain 应用中集成 `memory` 功能。
> 
> 如果你的代码已经在使用 `RunnableWithMessageHistory` 或 `BaseChatMessageHistory`，你**无需进行任何修改**。我们暂时没有计划弃用此功能，因为它仍然非常适合用于简单聊天应用。任何使用 `RunnableWithMessageHistory` 的代码将继续按预期工作。
> 
> 详情请参阅 [如何迁移到 LangGraph Memory](https://langchain-doc.cn/versions/migrating_memory/)。

## 概览

我们将通过一个示例来展示如何设计并实现一个由 LLM 驱动的聊天机器人。
该聊天机器人能够与用户进行对话，并记住之前与 [聊天模型](https://langchain-doc.cn/concepts/chat_models) 的交互内容。

请注意，我们构建的这个聊天机器人仅使用语言模型进行对话。
如果你在寻找更高级的功能，可以参考以下相关概念：

*   [Conversational RAG](https://langchain-doc.cn/tutorials/qa_chat_history)：让聊天机器人可以基于外部数据源进行对话
*   [Agents](https://langchain-doc.cn/tutorials/agents)：构建能执行操作的聊天机器人

本教程将介绍基础部分，这对理解以上两个高级主题很有帮助。但如果你已经熟悉，可以直接跳转。

## 设置

### Jupyter Notebook

本指南（以及大多数其他文档）均使用 [Jupyter notebooks](https://jupyter.org/)，并假设读者也使用相同环境。Jupyter notebooks 非常适合学习如何操作 LLM 系统，因为在实践过程中经常会出现意外（输出异常、API 无响应等），而交互式环境能帮助你更好地理解系统运行原理。

本教程和其他教程最方便的运行方式都是在 Jupyter notebook 中完成。安装方法请参考 [安装指南](https://jupyter.org/install)。

### 安装依赖

本教程需要使用 `langchain-core` 和 `langgraph`。请确保你的 `langgraph` 版本不低于 `0.2.28`。

#### Pip

```shell
pip install langchain-core langgraph>0.2.27
```

#### Conda

```shell
conda install langchain-core langgraph>0.2.27 -c conda-forge
```

更多详情请参阅 [安装指南](../how_to/installation.html)。

### LangSmith

你在 LangChain 中构建的许多应用可能包含多个步骤和多次 LLM 调用。
随着应用变得越来越复杂，能够清晰地查看链条或代理内部的运行过程就变得尤为重要。
实现这一点的最佳方式就是使用 [LangSmith](https://smith.langchain.com)。

注册账户后，**（请前往 LangSmith 网站的 Settings -> API Keys 页面创建 API Key）**，然后设置以下环境变量以启用日志追踪：

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

或者，在 Notebook 环境中可以这样设置：

```python
import getpass
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## 快速上手

首先，我们学习如何单独使用语言模型。LangChain 支持多种可互换的语言模型——请选择你想使用的模型：

选择会话模型：

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

[更新模型调用方法](https://python.langchain.com/docs/tutorials/chatbot/#quickstart)

我们可以直接调用该模型。`ChatModel` 是 LangChain “Runnable” 的实例，这意味着它们提供了统一的调用接口。要简单调用模型，只需将消息列表传递给 `.invoke` 方法。

```python
from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hi! I'm Bob")])
```

（此处省略输出内容）

单独的模型没有任何“记忆”或上下文。如果我们接着问一个问题：

```python
model.invoke([HumanMessage(content="What's my name?")])
```

模型不会记得之前的消息——因为它没有保存对话历史。
为了让聊天机器人“记得”我们说过的话，我们需要把整个 [对话历史](https://langchain-doc.cn/concepts/chat_history) 一并传入模型。

```python
from langchain_core.messages import AIMessage
model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
```

AIMessage(content='Your name is Bob! How can I help you today, Bob?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 33, 'total_tokens': 47, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0705bf87c0', 'finish_reason': 'stop', 'logprobs': None}, id='run-34bcccb3-446e-42f2-b1de-52c09936c02c-0', usage_metadata={'input_tokens': 33, 'output_tokens': 14, 'total_tokens': 47, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})

---

现在我们可以看到模型返回了一个正确的响应！

这就是聊天机器人能够进行“对话式交互”的基本原理。

那么我们该如何更好地实现它呢？

---

## 消息持久化（Message persistence）

[LangGraph](https://langchain-ai.github.io/langgraph/) 实现了一个 **内置的持久化层（persistence layer）**，这使它非常适合用于支持多轮对话的聊天应用程序。

将我们的聊天模型封装进一个最小的 LangGraph 应用中，就可以自动保存消息历史，从而简化多轮对话类应用的开发。

LangGraph 自带一个简单的 **内存检查点系统（in-memory checkpointer）**，我们在下面的示例中将使用它。
详细内容可以查看其[官方文档](https://langchain-ai.github.io/langgraph/concepts/persistence/)，其中介绍了如何使用其他持久化后端（例如 SQLite 或 Postgres）。

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
# 定义一个新的图
workflow = StateGraph(state_schema=MessagesState)
# 定义调用模型的函数
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}
# 定义图中的（单个）节点
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
# 添加内存持久化
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

接下来我们需要创建一个 `config`，并在每次调用可运行对象时传入。
这个配置包含一些不直接属于输入的数据，但仍然很重要的信息。
在这个例子中，我们希望包含一个 `thread_id`（线程 ID），配置如下：

```python
config = {"configurable": {"thread_id": "abc123"}}
```

这样我们就可以在一个应用中同时支持多个会话线程，这是支持多用户聊天应用时的常见需求。

然后我们就可以调用这个应用：

```python
query = "Hi! I'm Bob."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # 输出包含了所有消息的状态
```

输出结果：

```
================================== [1m Ai Message [0m ==================================
Hi Bob! How can I assist you today?
```

接着我们继续对话：

```python
query = "What's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

输出结果：

```
================================== [1m Ai Message [0m ==================================
Your name is Bob! How can I help you today, Bob?
```

很好！现在我们的聊天机器人已经可以**记住我们说过的话**了。

如果我们修改 `config`，将其指向一个不同的 `thread_id`，就能看到一个“全新的对话”：

```python
config = {"configurable": {"thread_id": "abc234"}}
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

输出结果：

```
================================== [1m Ai Message [0m ==================================
I'm sorry, but I don't have access to personal information about you unless you've shared it in this conversation. How can I assist you today?
```

不过我们仍然可以随时回到原来的对话，因为我们已经将它**持久化存储在数据库**中了。

```python
config = {"configurable": {"thread_id": "abc123"}}
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```
==================================[1m Ai Message [0m==================================
Your name is Bob. What would you like to discuss today?
```

这就是我们如何支持一个可以与多个用户进行对话的聊天机器人！

---

目前为止，我们只是给模型添加了一个简单的持久化层。
接下来，我们可以通过添加 **prompt 模板**，让聊天机器人更加复杂和个性化。

---

## Prompt 模板（Prompt templates）

[Prompt Templates](https://langchain-doc.cn/concepts/prompt_templates) 可以将原始用户信息转化为 LLM 可以理解的格式。
在这个例子中，原始的用户输入只是一个消息，我们把它传递给 LLM。现在我们可以让它稍微复杂一些：
首先，添加一个系统消息（system message），包含一些自定义指令（但仍然以 messages 作为输入）。然后，我们可以在输入中加入除了消息之外的其他内容。

要添加系统消息，可以创建一个 `ChatPromptTemplate`。我们将使用 `MessagesPlaceholder` 来传递所有消息：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

现在，我们可以更新应用以使用这个模板：

```python
workflow = StateGraph(state_schema=MessagesState)
def call_model(state: MessagesState):
    # highlight-start
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    # highlight-end
    return {"messages": response}
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

调用应用的方法与之前相同：

```python
config = {"configurable": {"thread_id": "abc345"}}
query = "Hi! I'm Jim."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```
==================================[1m Ai Message [0m==================================
Ahoy there, Jim! What brings ye to these waters today? Be ye seekin' treasure, knowledge, or perhaps a good tale from the high seas? Arrr!
```

```python
query = "What is my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

```
==================================[1m Ai Message [0m==================================
Ye be called Jim, matey! A fine name fer a swashbuckler such as yerself! What else can I do fer ye? Arrr!
```

太棒了！
现在让我们让 prompt 模板稍微复杂一些。假设新的 prompt 模板如下：

```python
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

注意，我们在 prompt 中添加了一个新的 `language` 输入。
此时，我们的应用有两个参数——输入的 `messages` 和 `language`。
我们需要更新应用的状态以反映这一变化。

```python
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
# highlight-next-line
class State(TypedDict):
    # highlight-next-line
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # highlight-next-line
    language: str
workflow = StateGraph(state_schema=State)
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

---

```python
config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Spanish"
input_messages = [HumanMessage(query)]
output = app.invoke(
    # highlight-next-line
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

```
==================================[1m Ai Message [0m==================================
¡Hola, Bob! ¿Cómo puedo ayudarte hoy?
```

请注意，整个状态都会被持久化，因此如果不想更改某些参数（比如 `language`），可以省略它们：

```python
query = "What is my name?"
input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()
```

```
==================================[1m Ai Message [0m==================================
Tu nombre es Bob. ¿Hay algo más en lo que pueda ayudarte?
```

要理解内部发生了什么，可以查看 [LangSmith trace](https://smith.langchain.com/public/15bd8589-005c-4812-b9b9-23e74ba4c3c6/r)。

---

## 管理对话历史（Managing Conversation History）

构建聊天机器人时，一个重要概念是如何管理对话历史。如果不加管理，消息列表会无限增长，并可能溢出 LLM 的上下文窗口（context window）。因此，需要增加一个步骤来限制传入消息的大小。

**重要：你需要在 prompt 模板之前，但在从消息历史中加载先前消息之后执行这个步骤。**

我们可以在 prompt 之前添加一个步骤，修改 `messages` 键，然后将这个新链包装在 Message History 类中。

LangChain 提供了一些内置的辅助工具来 [管理消息列表](../how_to/index.html#messages)。
在这个例子中，我们使用 [trim_messages](https://langchain-doc.cn/how_to/trim_messages/) 来减少发送给模型的消息数量。该工具允许你指定保留的 token 数量，以及其他参数，例如是否总是保留系统消息，是否允许部分消息：

```python
from langchain_core.messages import SystemMessage, trim_messages
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
trimmer.invoke(messages)
```

```
[SystemMessage(content="you're a good assistant", additional_kwargs={}, response_metadata={}),
 HumanMessage(content='whats 2 + 2', additional_kwargs={}, response_metadata={}),
 AIMessage(content='4', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='thanks', additional_kwargs={}, response_metadata={}),
 AIMessage(content='no problem!', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='having fun?', additional_kwargs={}, response_metadata={}),
 AIMessage(content='yes!', additional_kwargs={}, response_metadata={})]
```

要在链中使用它，只需在将 `messages` 输入传给 prompt 之前运行 trimmer。

```python
workflow = StateGraph(state_schema=State)
def call_model(state: State):
    print(f"Messages before trimming: {len(state['messages'])}")
    # highlight-start
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    # highlight-end
    return {"messages": [response]}
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

---

现在，如果我们问模型“我的名字是什么”，它可能不知道，因为我们裁剪了聊天历史（通过将 trim 策略设为 `'last'`，只保留最近的消息以符合 `max_tokens`）：

```python
config = {"configurable": {"thread_id": "abc567"}}
query = "What is my name?"
language = "English"
# highlight-next-line
input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

```
Messages before trimming: 12
Messages after trimming: 8
Remaining messages:
  SystemMessage: you're a good assistant
  HumanMessage: whats 2 + 2
  AIMessage: 4
  HumanMessage: thanks
  AIMessage: no problem!
  HumanMessage: having fun?
  AIMessage: yes!
  HumanMessage: What is my name?
==================================[1m Ai Message [0m==================================
I don't know your name. If you'd like to share it, feel free!
```

但是，如果我们询问最近消息中的信息，它仍然会记得：

```python
config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem was asked?"
language = "English"
input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
```

```
Messages before trimming: 12
Messages after trimming: 8
Remaining messages:
  SystemMessage: you're a good assistant
  HumanMessage: whats 2 + 2
  AIMessage: 4
  HumanMessage: thanks
  AIMessage: no problem!
  HumanMessage: having fun?
  AIMessage: yes!
  HumanMessage: What math problem was asked?
==================================[1m Ai Message [0m==================================
The math problem that was asked was "what's 2 + 2."
```

如果查看 LangSmith，你可以在 [LangSmith trace](https://smith.langchain.com/public/04402eaa-29e6-4bb1-aa91-885b730b6c21/r) 中看到底层发生了什么。

## 流式传输（Streaming）

现在我们已经有一个可以运行的聊天机器人了。然而，对于聊天机器人应用来说，有一个**非常重要的用户体验（UX）考虑**：那就是**流式输出（streaming）**。

因为大型语言模型（LLM）的响应可能需要一段时间才能生成，为了提升用户体验，大多数聊天类应用会采用“边生成边输出”的方式，即当模型生成每个 token（文字片段）时，就立即返回给用户显示。这样，用户可以**实时看到模型思考的过程**。

在 LangChain 中，这种流式输出实现起来非常简单——几乎与普通调用一样。让我们看看如何让聊天机器人支持流式传输。

```python
from langchain_core.messages import HumanMessage
config = {"configurable": {"thread_id": "streaming_123"}}
query = "Write a short poem about the ocean."
input_messages = [HumanMessage(query)]
```

### 普通调用（非流式）

```python
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

模型会等待全部生成完毕后一次性返回结果。

---

### 使用流式输出

如果我们想在生成过程中逐步接收输出，只需要调用 `.stream()` 方法：

```python
for chunk in app.stream({"messages": input_messages}, config):
    print(chunk)
```

你会看到模型逐步输出每一部分内容，比如：

```
{'messages': [AIMessageChunk(content='The ')]}
{'messages': [AIMessageChunk(content='ocean ')]}
{'messages': [AIMessageChunk(content='is ')]}
...
```

在实际应用中，你可以将这些输出实时显示到前端界面，从而让用户获得“实时回复”的体验，就像 ChatGPT 一样。

---

### 异步流式支持

如果你正在使用异步框架（如 FastAPI 或 Jupyter Notebook 的异步环境），LangChain 也提供了异步版本的流式调用：

```python
async for chunk in app.astream({"messages": input_messages}, config):
    print(chunk)
```

这在高并发场景或实时交互式应用中非常有用。

---

## 总结

我们刚刚一步步构建了一个可以**记忆上下文、支持多轮对话、具有多线程持久化**并且**支持流式响应**的聊天机器人。

你现在已经掌握了：

✅ 如何使用 LangChain 的 ChatModel 构建对话
✅ 如何通过 LangGraph 添加持久化内存
✅ 如何使用 Prompt Template 定义个性化风格
✅ 如何管理会话历史防止上下文溢出
✅ 如何使用 Streaming 提升聊天体验

这些是构建任何智能聊天系统的核心能力。接下来，你可以在此基础上进一步扩展，比如：

*   集成外部数据库（Postgres、Redis）保存会话
*   使用 LangGraph 的 **branching & conditional logic**（条件逻辑分支）
*   添加工具调用（Tool calling）让机器人执行操作
*   构建多角色对话系统（Multi-agent systems）

---

### 更多资源：

*   📘 [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
*   🧠 [LangChain Chat Models 概念](https://langchain-doc.cn/concepts/chat_models)
*   🧩 [LangGraph Memory（记忆系统）概念](https://langchain-ai.github.io/langgraph/concepts/persistence/)
*   🧭 [LangSmith 调试与可视化平台](https://smith.langchain.com)