import yaml
from loguru import logger

def load_yaml_config():
    """读取项目目录下的 config.yaml；文件缺失或格式错误时返回 None。
    注意这里用的是相对路径，服务需从项目目录启动。"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("配置文件 config.yaml 未找到。请确保文件存在于项目根目录。")
        return None
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {e}")
        return None

config = load_yaml_config()

# ---------- 必要配置 ----------
api_key = config.get('api_key') if config else None
base_url = config.get('base_url') if config else None
model_name = config.get('model_name') if config else None

# ---------- 可选配置：不写就是 None，由调用方按各自默认值处理 ----------
temperature = config.get('temperature') if config else None                      # 采样温度（不写 = 服务商默认）
send_temperature = config.get('send_temperature') if config else None            # 是否发送 temperature（不写 = true）
max_tokens = config.get('max_tokens') if config else None                        # 输出长度上限（不写 = 服务商默认）
send_max_tokens = config.get('send_max_tokens') if config else None              # 是否发送长度参数（不写 = true）
max_tokens_param = config.get('max_tokens_param') if config else None            # 长度参数名：max_tokens / max_completion_tokens（不写 = max_tokens）
json_mode = config.get('json_mode') if config else None                          # 是否要求返回 JSON（不写 = true）
thinking = config.get('thinking') if config else None                            # 是否开启思考（不写 = false；需 send_thinking 为 true 才发出）
send_thinking = config.get('send_thinking') if config else None                  # 是否发送 thinking（不写 = true，形式 {"thinking": {"type": ...}}）
extra_body_json = config.get('extra_body_json') if config else None              # 服务商特有参数，JSON 字符串（不写 = 不添加）
timeout = config.get('timeout') if config else None                              # 单次请求超时秒数（不写 = 60）
max_retries = config.get('max_retries') if config else None                      # SDK 自动重试次数：0/不写 = 关闭，2 = 开启