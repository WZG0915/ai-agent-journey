import requests

def get_weather(city):
    """
    封装的天气查询工具函数
    :param city: 城市英文名称，例如 'Beijing', 'Shanghai'
    """
    # 1. 构造 URL ?format=j1 是 wttr.in 的暗号，代表要求返回标准的 JSON 数据
    url = f"https://wttr.in/{city}?format=j1"

    # 2. 使用安全气囊，防止网络报错导致程序猝死
    try:
        #发送 GET 请求
        response = requests.get(url)
        #如果状态不是200，主动报警
        response.raise_for_status()

        # 3. 解析 JSON 响应为 Python 字典
        weather_data = response.json()

        # 4. 层层剥开 JSON 提取干货
        # 提取摄氏度温度
        current_temp = weather_data['current_condition'][0]['temp_C']
        # 提取天气描述
        weather_desc = weather_data['current_condition'][0]['weatherDesc'][0]['value']

        # 组装成一个好看的字符串返回
        return f"城市：{city} | 当前温度：{current_temp} | 天气状况：{weather_desc}"
    except Exception as e:
        return f"查询【{city}】天气时发生异常: {e}"
#-----测试调用get_weather()函数 -------
print("开始联网查询天气...")
result = get_weather("Beijing")
print(result)
