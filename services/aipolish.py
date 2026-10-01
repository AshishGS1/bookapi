from google import genai

from config import settings
from resources.prompts import ANS_PROMPT_TEMPLATE

_client = genai.Client(api_key=settings.gemini_api)


def generate_answer(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = ANS_PROMPT_TEMPLATE.format(context=context, query=query)

    chat = _client.chats.create(model=settings.gemini_model)
    response = chat.send_message(message=prompt)
    return response.text
