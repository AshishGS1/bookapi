from google import genai

from config import settings

_client = genai.Client(api_key=settings.gemini_api)

_PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context doesn't contain enough information to answer, say so — don't guess.

Context:
{context}

Question: {query}"""


def generate_answer(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = _PROMPT_TEMPLATE.format(context=context, query=query)

    chat = _client.chats.create(model=settings.gemini_model)
    response = chat.send_message(message=prompt)
    return response.text
