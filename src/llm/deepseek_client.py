import os
import sys
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- 修改开始: 动态计算绝对路径 ---
# 获取当前文件 (deepseek_client.py) 的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 回退两层找到项目根目录 (src/llm -> src -> root)
project_root = os.path.dirname(os.path.dirname(current_dir))
# 拼接 secrets.env 的绝对路径
env_path = os.path.join(project_root, "config", "secrets.env")

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print(f"⚠️ 警告: 未找到配置文件 {env_path}")
# --- 修改结束 ---

class DeepSeekClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com" 
        
        if not self.api_key:
            # 尝试从系统环境变量获取，作为 fallback
            self.api_key = os.getenv("DEEPSEEK_API_KEY_SYSTEM")
            
        if not self.api_key:
            raise ValueError(f"未找到 DEEPSEEK_API_KEY。请检查 {env_path} 文件。")

    def get_llm(self, temperature: float = 0.1, streaming: bool = True):
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            temperature=temperature,
            streaming=streaming,
            max_tokens=4096 
        )