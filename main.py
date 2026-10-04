from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from openai_llm import OpenaiLlm
from pathlib import Path
import sys
from loguru import logger
from settings import api_key, base_url, model_name

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} - {level: <8} - {message}"
)

missing = [
    name
    for name, value in (("model_name", model_name), ("base_url", base_url), ("api_key", api_key))
    if not str(value or "").strip()
]

if missing:
    print("=" * 46)
    print("  翻译服务无法启动")
    print("  config.yaml 中以下必填项尚未填写：" + "、".join(missing))
    print("  请填写后重新运行 run.bat。")
    print("=" * 46)
    raise SystemExit(1)

app = FastAPI()
client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)


def read_prompt_file(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return [{"role": "system", "content": content.strip()}]


system_messages = read_prompt_file(
    file_path=Path(__file__).parent / "prompt.txt"
)


async def do_translate(text: str, source: str, target: str) -> str:
    logger.info(f"翻译: {source} → {target}, 文本: {text}")
    openai_service = OpenaiLlm(
        system_messages=system_messages,
        model_name=model_name,
        client=client
    )
    result = await openai_service.call_openai_json_api(
        user_message=text,
        source_lang=source,
        target_lang=target
    )
    logger.success(f"翻译完成: {result}")
    return result["translatedText"]


@app.post("/")
async def translate_root(request: Request):
    try:
        body = await request.json()
        logger.info(f"收到请求(根路径): {body}")
        text = body.get("q")
        source = body.get("source", "auto")
        target = body.get("target", "zh")
        if not text:
            raise HTTPException(status_code=400, detail="Missing q field")
        translated = await do_translate(text, source, target)
        return JSONResponse(content={"translatedText": translated})
    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate_standard(request: Request):
    try:
        body = await request.json()
        logger.info(f"收到请求(/translate): {body}")
        text = body.get("q")
        source = body.get("source", "auto")
        target = body.get("target", "zh")
        if not text:
            raise HTTPException(status_code=400, detail="Missing q field")
        translated = await do_translate(text, source, target)
        return JSONResponse(content={"translatedText": translated})
    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )