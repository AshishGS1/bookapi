# query prefix for bge model
EMBED_INSTRUCTION= "Represent this sentence for searching relevant passages: "

#default prompt template to send to gemini
ANS_PROMPT_TEMPLATE= """Answer the question using only the context below. \
If the context doesn't contain enough information to answer, say so — don't guess.

Context:
{context}

Question: {query}"""