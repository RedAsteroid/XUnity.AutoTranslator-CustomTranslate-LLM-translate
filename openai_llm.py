"""OpenAI 兼容的翻译调用封装，不绑定 DeepSeek。

任何 OpenAI Chat Completions 端点都能用（DeepSeek / OpenAI / Kimi / 通义 / 智谱 /
OpenRouter / SiliconFlow / vLLM / Ollama / OneAPI 等），改 config.yaml 即可。
厂商差异由 config.yaml 的开关表达（send_temperature / send_max_tokens / send_thinking /
max_tokens_param / extra_body_json），代码不做厂商判断；参数被端点拒绝时自动降级重试。

可选配置（不写就用默认值，main.py 无需改动）：
    temperature: 0.3                # 采样温度（取值范围由服务商决定）
    send_temperature: true          # 是否发送 temperature
    max_tokens: 512                 # 输出长度上限
    send_max_tokens: true           # 是否发送长度参数
    max_tokens_param: max_tokens    # 或 max_completion_tokens
    json_mode: true                 # 是否要求返回 JSON
    thinking: false                 # 是否开启思考
    send_thinking: true             # 是否发送 thinking（{"thinking": {"type": ...}}）
    extra_body_json: ''             # 服务商特有参数（JSON 字符串）
    timeout: 60
    max_retries: 0                  # SDK 自动重试次数，0 = 关闭
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from openai import AsyncOpenAI, BadRequestError

# settings.py 的可选开关，读不到就用默认值
try:
    import settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default: Any = None) -> Any:
    """读取 settings.py 的可选配置；缺失或为 None 都回落默认值。"""
    if _settings is None:
        return default
    value = getattr(_settings, name, None)
    return default if value is None else value

def _cfg_int(name: str, default: int) -> int:
    """读取 settings.py 的整数项，缺失或非法一律回落默认值。"""
    try:
        return int(_cfg(name, default))
    except (TypeError, ValueError):
        return default

class OpenaiLlm:
    # 400 报错里出现这些字样 = 端点不支持该可选参数，而非请求有误
    _OPTIONAL_PARAM_HINTS = ("response_format", "temperature", "max_tokens", "max_completion_tokens",
                             "thinking", "enable_thinking", "reasoning_effort", "json_object")
    # 这些是确定性失败、与可选参数无关的错误，命中就不重试
    _UNRELATED_ERROR_HINTS = ("context length", "context_length", "too long", "model not exist",
                              "no such model", "does not exist", "insufficient", "quota", "balance")
    
    def __init__(
        self,
        system_messages: Union[str, List[Dict[str, str]]],
        model_name: str,
        client: AsyncOpenAI,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        json_mode: Optional[bool] = None,
        thinking: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        send_temperature: Optional[bool] = None,
        send_max_tokens: Optional[bool] = None,
        send_thinking: Optional[bool] = None,
        max_tokens_param: Optional[str] = None,
        extra_body_json: Optional[str] = None,
    ):
        self.system_messages = system_messages
        self.model_name = model_name or ""
        self.client = client

        # 显式参数 > settings.py > 默认值
        self.temperature = temperature if temperature is not None else _cfg("temperature")
        self.max_tokens = max_tokens if max_tokens is not None else _cfg("max_tokens")
        self.timeout = timeout if timeout is not None else (_cfg("timeout") or 60.0)
        self.json_mode = bool(json_mode) if json_mode is not None else bool(_cfg("json_mode", True))
        self.thinking = bool(thinking) if thinking is not None else bool(_cfg("thinking", False))
        self.extra_body = dict(extra_body or {})
        self.max_retries = max_retries if max_retries is not None else _cfg_int("max_retries", 0)
        # 三个发送开关：为 false 时对应参数完全不进请求体；默认 true
        self.send_temperature = bool(send_temperature) if send_temperature is not None else bool(_cfg("send_temperature", True))
        self.send_max_tokens = bool(send_max_tokens) if send_max_tokens is not None else bool(_cfg("send_max_tokens", True))
        self.send_thinking = bool(send_thinking) if send_thinking is not None else bool(_cfg("send_thinking", True))
        self.max_tokens_param = str(max_tokens_param or _cfg("max_tokens_param") or "max_tokens")
        if self.max_tokens_param not in ("max_tokens", "max_completion_tokens"):
            self.max_tokens_param = "max_tokens"
        raw_extra = extra_body_json if extra_body_json is not None else _cfg("extra_body_json")
        if raw_extra:
            try:
                parsed = json.loads(raw_extra) if isinstance(raw_extra, str) else dict(raw_extra)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                self.extra_body.update(parsed)
            else:
                logger.warning("extra_body_json 不是合法的 JSON 对象，已忽略")

    # ---------------------------------------------------------- 400 判定
    @classmethod
    def _mentions_optional_params(cls, error: Exception) -> bool:
        text = str(error).lower()
        if any(hint in text for hint in cls._UNRELATED_ERROR_HINTS):
            return False
        return any(hint in text for hint in cls._OPTIONAL_PARAM_HINTS)

    # ---------------------------------------------------------- 消息构造
    def _system_text(self) -> str:
        """兼容 str 与 [{"role": "system", "content": ...}, ...]。"""
        source = self.system_messages
        if isinstance(source, str):
            return source
        parts: List[str] = []
        for item in source or []:
            if isinstance(item, dict):
                parts.append(item.get("content") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(part for part in parts if part)

    def build_messages(self, user_message, source_lang="auto", target_lang="zh"):
        system_content = self._system_text()
        system_content = system_content.replace("{source_lang}", str(source_lang))
        system_content = system_content.replace("{target_lang}", str(target_lang))
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

    # ---------------------------------------------------------- 请求参数
    def _optional_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        if self.send_temperature and self.temperature is not None:
            params["temperature"] = float(self.temperature)
        if self.send_max_tokens and self.max_tokens:
            params[self.max_tokens_param] = int(self.max_tokens)
        body = dict(self.extra_body)
        if self.send_thinking:
            body.setdefault("thinking", {"type": "enabled" if self.thinking else "disabled"})
        if body:
            params["extra_body"] = body

        return params

    # ---------------------------------------------------------- 实际调用
    def _call_client(self) -> AsyncOpenAI:
        """按 max_retries 重建客户端：默认 0（关闭 SDK 自动重试），配合降级重试即最多 2 次尝试。"""
        try:
            return self.client.with_options(max_retries=self.max_retries)
        except Exception:
            return self.client

    async def _request(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "timeout": self.timeout,
        }
        payload.update(params)
        response = await self._call_client().chat.completions.create(**payload)
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        content = getattr(response.choices[0].message, "content", None)
        if isinstance(content, str):
            return content.strip()
        # 少数网关按分片返回
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(item.get("text") or "")
                else:
                    chunks.append(getattr(item, "text", "") or "")
            return "".join(chunks).strip()
        return ""

    def _error_message(self, error: Exception) -> str:
        return (
            f"调用模型 {self.model_name} 失败"
            f"（base_url={self.client.base_url}，请检查 model_name / base_url / api_key "
            f"是否匹配该端点）: {error}"
        )

    async def call_openai_json_api(
        self, user_message, source_lang="auto", target_lang="zh"
    ) -> Dict[str, Any]:
        messages = self.build_messages(
            user_message=user_message,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        params = self._optional_params()

        try:
            content = await self._request(messages, params)
        except BadRequestError as error:
            # 端点不认识某个可选参数时（response_format / temperature / 长度参数 / thinking 等），
            # 去掉全部可选参数重试一次
            if not params or not self._mentions_optional_params(error):
                raise RuntimeError(self._error_message(error)) from error
            logger.warning(f"端点不接受可选参数，去掉后重试一次: {error}")
            try:
                content = await self._request(messages, {})
            except Exception as retry_error:
                raise RuntimeError(self._error_message(retry_error)) from retry_error
        except Exception as error:
            raise RuntimeError(self._error_message(error)) from error

        return self.parse_json(content)

    # ---------------------------------------------------------- JSON 兜底解析
    @classmethod
    def parse_json(cls, content: str) -> Dict[str, Any]:
        raw = content or ""
        text = cls._strip_code_fence(raw)

        data: Any = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # JSON 前后夹了说明文字时，截取首尾花括号再试
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict):
            raise ValueError(f"AI响应不是有效的JSON对象（前200字符）: {raw[:200]!r}")

        if not isinstance(data.get("translatedText"), str):
            # 兼容其他模型的字段名，最后退化为「唯一的字符串值」
            for key in ("translated_text", "translation", "translated", "text", "result", "output", "content"):
                value = data.get(key)
                if isinstance(value, str):
                    data = {**data, "translatedText": value}
                    break
            else:
                strings = [v for v in data.values() if isinstance(v, str)]
                if len(strings) == 1:
                    data = {**data, "translatedText": strings[0]}
                else:
                    raise ValueError(f"AI响应JSON里找不到 translatedText 字段: {raw[:200]!r}")

        return data

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()