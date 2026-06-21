"""Lớp trung gian gọi LLM. Các module khác chỉ gọi invoke_llm, không cần biết backend."""
from functools import lru_cache

from langchain_core.messages import HumanMessage

from src.config import settings


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise RuntimeError("Thiếu GOOGLE_API_KEY trong .env để dùng Gemini.")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_hf_local():
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model)
    gen = pipeline("text-generation", model=model, tokenizer=tok, return_full_text=False, max_new_tokens=1024)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))


@lru_cache(maxsize=2)
def get_llm(provider: str | None = None):
    provider = provider or settings.llm_provider
    if provider == "gemini":
        return _build_gemini()
    if provider == "hf_local":
        return _build_hf_local()
    raise ValueError(f"llm_provider không hỗ trợ: '{provider}'")


def invoke_llm(prompt: str, provider: str | None = None) -> str:
    resp = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)
