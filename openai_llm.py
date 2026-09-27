from openai import AsyncOpenAI
import json


class OpenaiLlm:
    def __init__(self, system_messages, model_name, client: AsyncOpenAI):
        self.system_messages = system_messages
        self.model_name = model_name
        self.client = client

    def build_messages(self, user_message, source_lang="auto", target_lang="zh"):
        system_content = self.system_messages[0]["content"]
        system_content = system_content.replace("{source_lang}", source_lang)
        system_content = system_content.replace("{target_lang}", target_lang)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message}
        ]
        return messages

    async def call_openai_json_api(self, user_message, source_lang="auto", target_lang="zh"):
        try:
            if "deepseek" in self.model_name:
                response_format = {"type": "json_object"}
                extra = {"thinking": {"type": "disabled"}}   # 关闭思考模式
            else:
                response_format = None
                extra = None

            messages = self.build_messages(
                user_message=user_message,
                source_lang=source_lang,
                target_lang=target_lang
            )

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                response_format=response_format,
                timeout=60.0,
                extra_body=extra,
            )

            ai_response_json = response.choices[0].message.content
            return json.loads(ai_response_json)

        except json.JSONDecodeError as e:
            raise ValueError(f"AI响应不是有效的JSON格式: {e}")
        except Exception as e:
            raise RuntimeError(f"调用OpenAI API时出现错误: {e}")