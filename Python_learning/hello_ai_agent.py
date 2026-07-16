'''
脚本功能：
    从 .env 文件中加载 DeepSeek API 密钥，
    调用 DeepSeek 的大模型要求输出 "Hello, AI Agent!" 并打印。
'''

# requests调用大模型
'''
import os 
from dotenv import load_dotenv
import requests

# 1. 加载.env中的环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError(
        "未找到DEEPSEEK_API_KEY环境变\n"
        "请确保 .env文件中添加了一行：DEEPSEEK_API_KEY=你的密钥"
    )


# 2. 定义请求的 URL 和 Headers
url = "https://api.deepseek.com/chat/completions"
# HTTP 头部，和 OpenAI 一样使用 Bearer 认证
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 3. 构造请求体(payload)
payload = {
    "model": "deepseek-v4-flash",
    "messages":[
        {
            "role": "user",
            "content": "请对我说‘Hello, AI Agent!’，不要包含任何其他多余的解释或字句。"
        }
    ],
    "temperature": 0.0 
}

print("正在使用最新的官方规范发送请求...")
# 4. 发送请求并处理响应
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()     #检测 HTTP 状态码
    if response.status_code == 200:
        response_dict = response.json()     #解析JSON相应
        reply = response_dict['choices'][0]['message']['content']
        print("\n---大模型回复---")
        print(reply)
        print("-----------------")
    else:
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
except Exception as e:
    print(f"网络请求发生异常: {e}")
'''


'''
    与 requests 版本的区别：
        - 无需手动构造 URL、Headers、JSON 负载
        - 由 openai 库处理认证、重试、流式等逻辑
        - 代码更简洁、更符合生产环境最佳实践
'''

#SDK调用大模型
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载.env 中的环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError(
        "未找到 DEEPSEEK_API_KEY 环境变量\n"
        "请确保 .env 中添加了一行：DEEPSEEK_API_KEY = 你的密钥"
    )


# 2. 创建 DeepSeek 客户端（完全兼容 OpenAI SDK）
#    base_url 指向 DeepSeek 的 API 根地址，SDK 会自动拼接 /v1/chat/completions
client = OpenAI(
    api_key = api_key,
    base_url = "https://api.deepseek.com",   # 官方指定的 Base URL
)

# 3. 调用聊天补全接口
#    这里完全采用 OpenAI 的调用风格，参数名与 requests 版本的 payload 一致
print("正在使用 DeepSeek SDK（OpenAI 兼容）发送请求...")

try:
    response = client.chat.completions.create(
        model = "deepseek-v4-flash",
        messages = [
            {
                "role": "user",
                "content": "请对我说‘Hello, AI Agent!’，不要包含任何其他多余的解释或字句。" 
            }
        ],
        temperature = 0.0
    )
    # 4. 提取回复文本
    # response 是一个 ChatCompletion 对象，结构类似 requests 版本的 JSON
    #    可以直接通过属性访问，不需要手动解析
    reply = response.choices[0].message.content
    print("\n---大模型回复---")
    print(reply)
    print("-----------------")

except Exception as e:
    print(f"SDK 调用发生异常: {e}")