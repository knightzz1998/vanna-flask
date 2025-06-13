# 导入 os 模块，用于与操作系统进行交互
import os

# 从上级目录的 base 模块中导入 VannaBase 类
from vanna.base import VannaBase
from openai import OpenAI
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

# 定义一个名为 Mistral 的类，继承自 VannaBase 类
class CustomeLLM(VannaBase):
    # 类的构造函数，用于初始化 Mistral 客户端
    def __init__(self, config=None):
        # 检查配置是否为 None，如果是则抛出异常
        if config is None:
            raise ValueError(
                "For Mistral, config must be provided with an api_key and model"
            )

        # 检查配置中是否包含 api_key，如果不包含则抛出异常
        if "api_key" not in config:
            raise ValueError("config must contain a CustomeLLM api_key")

        # 检查配置中是否包含 model，如果不包含则抛出异常
        if "model" not in config:
            raise ValueError("config must contain a CustomeLLM model")

        if "base_url" not in config:
            raise ValueError("config must contain a CustomeLLM base_url")

        # 从配置中获取 api_key
        api_key = config["api_key"]
        # 从配置中获取 model
        model = config["model"]
        # 从配置中获取 base_url
        base_url = config["base_url"]
        # 初始化 Mistral 客户端
        self.client = OpenAI(api_key=api_key,
                             base_url=base_url)
        # 将模型名称存储在类的属性中
        self.model = model

    # 生成系统消息的方法
    def system_message(self, message: str) -> any:
        # 返回一个包含系统角色和消息内容的字典
        return {"role": "system", "content": message}

    # 生成用户消息的方法
    def user_message(self, message: str) -> any:
        # 返回一个包含用户角色和消息内容的字典
        return {"role": "user", "content": message}

    # 生成助手消息的方法
    def assistant_message(self, message: str) -> any:
        # 返回一个包含助手角色和消息内容的字典
        return {"role": "assistant", "content": message}

    # 生成 SQL 语句的方法
    def generate_sql(self, question: str, **kwargs) -> str:
        # 调用父类的 generate_sql 方法生成 SQL 语句
        sql = super().generate_sql(question, **kwargs)

        # 将 SQL 语句中的 "\_" 替换为 "_"
        sql = sql.replace("\\_", "_")

        # 返回处理后的 SQL 语句
        return sql

    # 提交提示词并获取响应的方法
    def submit_prompt(self, prompt, **kwargs) -> str:
        # 调用 Mistral 客户端的 chat.complete 方法获取聊天响应
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt
        )

        # 返回聊天响应中第一个选择的消息内容
        return chat_response.choices[0].message.content

class MyVanna(ChromaDB_VectorStore, CustomeLLM):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        CustomeLLM.__init__(self, config=config)