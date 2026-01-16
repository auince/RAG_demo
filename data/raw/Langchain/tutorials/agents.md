# 构建一个代理（Agent）

LangChain 支持创建 [代理](https://langchain-doc.cn/concepts/agents)，即使用 [LLMs](https://langchain-doc.cn/concepts/chat_models) 作为推理引擎来决定执行哪些操作以及执行这些操作所需的输入的系统。 在执行操作后，结果可以反馈回 LLM，以确定是否需要进一步操作，或者是否可以结束。这通常通过 [调用工具](https://langchain-doc.cn/concepts/tool_calling) 来实现。

在本教程中，我们将构建一个可以与搜索引擎交互的代理。你可以向这个代理提问，观察它调用搜索工具，并与它进行对话。

## 端到端代理

下面的代码片段展示了一个完整功能的代理，它使用 LLM 来决定使用哪些工具。该代理配备了通用搜索工具，并具有会话记忆——意味着它可以作为多轮聊天机器人使用。

在本指南的后续部分，我们将逐步讲解各个组件及其功能——但如果你只想获取一些代码并开始使用，也可以直接使用它！

```python
# 导入相关功能
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
# 创建代理
memory = MemorySaver()
model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
search = TavilySearch(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)
```

```python
# 使用代理
config = {"configurable": {"thread_id": "abc123"}}
input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```
================================[1m 用户消息 [0m=================================
Hi, I'm Bob and I live in SF.
==================================[1m AI 消息 [0m==================================
你好，Bob！我注意到你介绍了自己并提到你住在旧金山（San Francisco），但你没有提出具体问题或需要使用工具的请求。你是否想了解关于旧金山或其他主题的某些信息？我可以使用可用的搜索工具帮助你查找相关信息。
```

```python
input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```
================================[1m 用户消息 [0m=================================
What's the weather where I live?
==================================[1m AI 消息 [0m==================================
[{'text': '让我搜索一下旧金山当前的天气信息。', 'type': 'text'}, {'id': 'toolu_011kSdheoJp8THURoLmeLtZo', 'input': {'query': 'current weather San Francisco CA'}, 'name': 'tavily_search', 'type': 'tool_use'}]
工具调用:
  tavily_search (toolu_011kSdheoJp8THURoLmeLtZo)
 调用 ID: toolu_011kSdheoJp8THURoLmeLtZo
  参数:
    query: current weather San Francisco CA
=================================[1m 工具消息 [0m=================================
名称: tavily_search
{"query": "current weather San Francisco CA", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Weather in San Francisco, CA", "url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1750168606, 'localtime': '2025-06-17 06:56'}, 'current': {'last_updated_epoch': 1750167900, 'last_updated': '2025-06-17 06:45', 'temp_c': 11.7, 'temp_f': 53.1, 'is_day': 1, 'condition': {'text': 'Fog', 'icon': '//cdn.weatherapi.com/weather/64x64/day/248.png', 'code': 1135}, 'wind_mph': 4.0, 'wind_kph': 6.5, 'wind_degree': 215, 'wind_dir': 'SW', 'pressure_mb': 1017.0, 'pressure_in': 30.02, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 86, 'cloud': 0, 'feelslike_c': 11.3, 'feelslike_f': 52.4, 'windchill_c': 8.7, 'windchill_f': 47.7, 'heatindex_c': 9.8, 'heatindex_f': 49.7, 'dewpoint_c': 9.6, 'dewpoint_f': 49.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 6.3, 'gust_kph': 10.2}}", "score": 0.944705, "raw_content": null}, {"title": "Weather in San Francisco in June 2025", "url": "https://world-weather.info/forecast/usa/san_francisco/june-2025/", "content": "Detailed ⚡ San Francisco Weather Forecast for June 2025 - day/night 🌡️ temperatures, precipitations - World-Weather.info. Add the current city. Search. Weather; Archive; Weather Widget °F. World; United States; California; Weather in San Francisco; ... 17 +64° +54° 18 +61° +54° 19", "score": 0.86441374, "raw_content": null}], "response_time": 2.34}
==================================[1m AI 消息 [0m==================================
根据搜索结果，以下是旧金山当前的天气：
- 温度：53.1°F（11.7°C）
- 天气：有雾
- 风速：来自西南方向，4.0 mph
- 湿度：86%
- 能见度：9 英里
这在旧金山是典型天气，该市以雾闻名。你想了解更多关于天气或旧金山的信息吗？
```

## 设置

### Jupyter Notebook

本指南（以及文档中的大多数其他指南）使用 [Jupyter notebooks](https://jupyter.org/)，并假设读者也在使用。 Jupyter notebooks 是学习如何使用 LLM 系统的理想交互环境，因为有时可能出现问题（意外输出、API不可用等），观察这些情况有助于更好地理解 LLM 构建过程。

本教程及其他教程最方便的运行方式是 Jupyter notebook。请参见 [这里](https://jupyter.org/install) 获取安装说明。

### 安装

安装 LangChain 运行：

```python
%pip install -U langgraph langchain-tavily langgraph-checkpoint-sqlite
```

更多详情请参阅我们的 [安装指南](../how_to/installation.html)。

### LangSmith

你使用 LangChain 构建的许多应用会包含多个步骤和多次 LLM 调用。 随着应用复杂度增加，检查链或代理内部发生了什么变得至关重要。 最佳方法是使用 [LangSmith](https://smith.langchain.com)。

在上述链接注册后，确保设置环境变量以开始记录追踪：

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

或者在 notebook 中使用：

```python
import getpass
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

### Tavily

我们将使用 [Tavily](https://langchain-doc.cn/integrations/tools/tavily_search)（搜索引擎）作为工具。 使用前，你需要获取并设置 API Key：

```bash
export TAVILY_API_KEY="..."
```

在 notebook 中可以使用：

```python
import getpass
import os
os.environ["TAVILY_API_KEY"] = getpass.getpass()
```

## 定义工具

首先，我们需要创建要使用的工具。主要工具是 [Tavily](https://langchain-doc.cn/integrations/tools/tavily_search) —— 一个搜索引擎。 可以使用专用 [langchain-tavily](https://pypi.org/project/langchain-tavily/) [集成包](https://langchain-doc.cn/concepts/architecture/#integration-packages) 将 Tavily 搜索引擎作为工具与 LangChain 一起使用。

```python
from langchain_tavily import TavilySearch
search = TavilySearch(max_results=2)
search_results = search.invoke("What is the weather in SF")
print(search_results)
# 如果需要，可以创建其他工具
# 一旦有了所有工具，我们可以将它们放入列表中以便后续引用
tools = [search]
```

## 使用语言模型

接下来，让我们学习如何使用语言模型调用工具。 LangChain 支持多种可互换使用的语言模型——选择你想用的模型即可。

```python
# | output: false
# | echo: false
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-latest")
```

你可以通过传入消息列表调用语言模型。默认情况下，响应是 `content` 字符串。

```python
query = "Hi!"
response = model.invoke([{"role": "user", "content": query}])
response.text()
```

```
'Hello! How can I help you today?'
```

现在我们可以让模型支持工具调用。为此，我们使用 `.bind_tools` 将工具知识绑定给语言模型。

```python
model_with_tools = model.bind_tools(tools)
```

接下来调用模型。首先使用普通消息，看它如何响应，同时观察 `content` 字段和 `tool_calls` 字段。

```python
query = "Hi!"
response = model_with_tools.invoke([{"role": "user", "content": query}])
print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")
```

```
消息内容: Hello! I'm here to help you. I have access to a powerful search tool that can help answer questions and find information about various topics. What would you like to know about?
随时提问或请求信息，我会尽力使用可用工具帮助你。
工具调用: []
```

现在尝试输入一个会触发工具调用的请求：

```python
query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])
print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")
```

```
消息内容: I'll help you search for information about the weather in San Francisco.
工具调用: [{'name': 'tavily_search', 'args': {'query': 'current weather San Francisco'}, 'id': 'toolu_015gdPn1jbB2Z21DmN2RAnti', 'type': 'tool_call'}]
```

可以看到，此时模型提示要调用 Tavily 搜索工具，但尚未实际调用。要真正调用工具，我们需要创建代理。

## 创建代理

定义工具和 LLM 后，我们可以创建代理。 我们将使用 [LangGraph](https://langchain-doc.cn/concepts/architecture/#langgraph) 构建代理。 当前使用的是高级接口，但 LangGraph 的高级接口背后是低级、可高度控制的 API，可修改代理逻辑。

初始化代理时传入 LLM 和工具。注意我们传入的是 `model` 而非 `model_with_tools`，因为 `create_react_agent` 会在内部调用 `.bind_tools`。

```python
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(model, tools)
```

## 运行代理

现在可以用几个查询来运行代理！注意，这里都是**无状态**查询（不会记住之前的交互）。 代理在交互结束时返回**最终状态**（包含所有输入，稍后可学习如何仅获取输出）。

首先，看不需要调用工具时的响应：

```python
input_message = {"role": "user", "content": "Hi!"}
response = agent_executor.invoke({"messages": [input_message]})
for message in response["messages"]:
    message.pretty_print()
```

```
================================[1m 用户消息 [0m=================================
Hi!
==================================[1m AI 消息 [0m==================================
你好！我可以使用可用搜索工具帮助你回答问题。请随意提问，我会尽力为你找到相关且准确的信息。
```

若想查看具体内部运作（确保未调用工具），可查看 [LangSmith trace](https://smith.langchain.com/public/28311faa-e135-4d6a-ab6b-caecf6482aaa/r)。

再尝试一个应调用工具的示例：

```python
input_message = {"role": "user", "content": "Search for the weather in SF"}
response = agent_executor.invoke({"messages": [input_message]})
for message in response["messages"]:
    message.pretty_print()
```

```
================================[1m 用户消息 [0m=================================
Search for the weather in SF
==================================[1m AI 消息 [0m==================================
[{'text': "I'll help you search for weather information in San Francisco. Let me use the search engine to find current weather conditions.", 'type': 'text'}, {'id': 'toolu_01WWcXGnArosybujpKzdmARZ', 'input': {'query': 'current weather San Francisco SF'}, 'name': 'tavily_search', 'type': 'tool_use'}]
工具调用:
  tavily_search (toolu_01WWcXGnArosybujpKzdmARZ)
 调用 ID: toolu_01WWcXGnArosybujpKzdmARZ
  参数:
    query: current weather San
```

Francisco SF
=================================[1m 工具消息 [0m=================================
名称: tavily_search

```
{"query": "current weather San Francisco SF", ... }
==================================[1m AI 消息 [0m==================================
根据搜索结果，以下是旧金山当前的天气：
- 温度：53.1°F（11.7°C）
- 天气：有雾
- 风速：4.0 mph，来自西南
- 湿度：86%
- 能见度：9.0 英里
- 体感温度：52.4°F（11.3°C）
天气似乎是旧金山典型天气，晨雾和温和气温。体感温度为 52.4°F（11.3°C）。
```

可查看 [LangSmith trace](https://smith.langchain.com/public/f520839d-cd4d-4495-8764-e32b548e235d/r) 以确认工具调用正常。

## 流式消息

除了使用 `.invoke` 获取最终响应，也可以流式返回消息，显示中间进度：

```python
for step in agent_executor.stream({"messages": [input_message]}, stream_mode="values"):
    step["messages"][-1].pretty_print()
```

```
================================[1m 用户消息 [0m=================================
Search for the weather in SF
==================================[1m AI 消息 [0m==================================
[{'text': "I'll help you search for information about the weather in San Francisco.", 'type': 'text'}, {'id': 'toolu_01DCPnJES53Fcr7YWnZ47kDG', 'input': {'query': 'current weather San Francisco'}, 'name': 'tavily_search', 'type': 'tool_use'}]
工具调用:
  tavily_search (toolu_01DCPnJES53Fcr7YWnZ47kDG)
 调用 ID: toolu_01DCPnJES53Fcr7YWnZ47kDG
  参数:
    query: current weather San Francisco
=================================[1m 工具消息 [0m=================================
名称: tavily_search
{"query": "current weather San Francisco", ... }
==================================[1m AI 消息 [0m==================================
根据搜索结果，以下是旧金山当前天气：
- 温度：53.1°F（11.7°C）
- 天气：有雾
- 风速：4.0 mph，来自西南
- 湿度：86%
- 能见度：9.0 英里
- 体感温度：52.4°F（11.3°C）
这是旧金山典型天气，尤其是早晨雾气。城市靠近海洋，地理特征独特，通常温和并伴有雾。
```

## 流式 token

除了流式返回消息，还可以流式返回 token，通过指定 `stream_mode="messages"` 实现。

```python
for step, metadata in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="messages"
):
    if metadata["langgraph_node"] == "agent" and (text := step.text()):
        print(text, end="|")
```

```
I|'ll help you search for information| about the weather in San Francisco.|Base|d on the search results, here|'s the current weather in| San Francisco:
-| Temperature: 53.1°F (|11.7°C)
-| Condition: Foggy
- Wind:| 4.0 mph from| the Southwest
- Humidity|: 86%|
- Visibility: 9|.0 miles
- Pressure: |30.02 in|Hg
...
```

## 添加记忆

如前所述，这个代理是无状态的，不会记住之前的交互。 要赋予记忆，需要传入 checkpointer，并在调用代理时提供 `thread_id`（以知道从哪个线程/会话继续）。

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
```

```python
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}
```

```python
for step in agent_executor.stream(
    {"messages": [("user", "Hi, I'm Bob!")]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```
================================[1m 用户消息 [0m=================================
Hi, I'm Bob!
==================================[1m AI 消息 [0m==================================
你好，Bob！我是一个 AI 助手，可以帮助你使用搜索工具查找信息。你想了解或搜索什么特定信息吗？我可以帮你找到准确、最新的资料。
```

```python
for step in agent_executor.stream(
    {"messages": [("user", "What is my name?")]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```
================================[1m 用户消息 [0m=================================
What's my name?
==================================[1m AI 消息 [0m==================================
你的名字是 Bob，就像你之前介绍的那样。我可以在对话中记住共享的信息，而无需再次搜索。
```

示例 [LangSmith trace](https://smith.langchain.com/public/fa73960b-0f7d-4910-b73d-757a12f33b2b/r)

若要开始新对话，只需更改 `thread_id`：

```python
config = {"configurable": {"thread_id": "xyz123"}}
for step in agent_executor.stream(
    {"messages": [("user", "What is my name?")]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
```

```
================================[1m 用户消息 [0m=================================
What's my name?
==================================[1m AI 消息 [0m==================================
抱歉，我无法访问任何能告诉我你名字的工具。我只能使用 tavily_search 函数搜索公开可用的信息。我无法获取用户的个人信息。如果你愿意告诉我名字，我会很高兴称呼你。
```

## 总结

到此为止！在本快速入门中，我们讲解了如何创建一个简单代理。 然后展示了如何流式返回响应——不仅包括中间步骤，也包括 token！ 我们还增加了记忆，使你可以与代理进行对话。 代理是一个复杂主题，需要深入学习！

更多关于代理的信息，请查看 [LangGraph](https://langchain-doc.cn/concepts/architecture/#langgraph) 文档，其中包含概念、教程和操作指南。