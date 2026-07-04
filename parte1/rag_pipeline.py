import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List, Any
from openai import OpenAI
from config.settings import (
    OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TOP_K
)

@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: List[Any]

class RAGPipeline:
    def __init__(self, retriever: Any, retriever_name: str = "dense", top_k: int = TOP_K):
        self.retriever = retriever
        self.retriever_name = retriever_name
        self.top_k = top_k
        self.llm = OpenAI(api_key=OPENAI_API_KEY)

    def answer(self, question: str, k: int = None) -> RAGResponse:
        k = k or self.top_k
        
        # 1. Recuperar los chunks como objetos (para los logs de la consola)
        chunks = self.retriever.retrieve(question, k=k)
        
        # 2. Recuperar el contexto como string formateado (para enviarlo al LLM)
        context = self.retriever.retrieve_as_context(question, k=k)
        
        # 3. Formatear el prompt
        user_msg = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # 4. Llamada a la API de OpenAI
        try:
            completion = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE
            )
            answer_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"[ERROR OpenAI] Falló la generación de respuesta: {e}")
            answer_text = "Error al generar la respuesta."
        
        return RAGResponse(
            question=question,
            answer=answer_text,
            retrieved_chunks=chunks
        )