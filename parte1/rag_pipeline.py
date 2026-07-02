"""
parte1/rag_pipeline.py  —  Parte 1.2
--------------------------------------
Pipeline RAG completo:
  1. Retriever (denso o cualquier otro que devuelva RetrievedChunk)
  2. Generación con GPT-4o-mini

El prompt del sistema y el modelo LLM son FIJOS en todas las partes;
lo único que cambia entre estrategias es el retriever que se inyecta.

Parámetros reportados:
  - k = 5 (ver config/settings.py TOP_K)
  - Prompt: ver config/settings.py SYSTEM_PROMPT y USER_PROMPT_TEMPLATE
  - Modelo LLM: gpt-4o-mini
  - Temperatura: 0.2 (respuestas factuales, baja variabilidad)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Instala openai:  pip install openai")

from config.settings import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    TOP_K,
)


# ── Protocolo de retriever (duck typing) ─────────────────────────────────────

@runtime_checkable
class Retriever(Protocol):
    """
    Cualquier retriever que implemente retrieve_as_context() puede usarse
    en el pipeline. Esto permite intercambiar Dense, Graph o Hybrid.
    """
    def retrieve_as_context(self, question: str, k: int | None = None) -> str:
        ...

    def retrieve(self, question: str, k: int | None = None) -> list:
        ...


# ── Respuesta estructurada ────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Respuesta completa del pipeline RAG."""
    question: str
    answer: str
    context: str
    retriever_name: str
    top_k: int
    retrieved_chunks: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def print_summary(self) -> None:
        """Imprime resumen legible por consola."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"RETRIEVER : {self.retriever_name}")
        print(f"PREGUNTA  : {self.question}")
        print(f"TOP-K     : {self.top_k}")
        print(f"TOKENS    : prompt={self.prompt_tokens} | completion={self.completion_tokens}")
        print(f"{'─'*70}")
        print("CONTEXTO RECUPERADO:")
        print(self.context)
        print(f"{'─'*70}")
        print("RESPUESTA:")
        print(self.answer)
        print(sep)


# ── Pipeline RAG ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Pipeline RAG modular:
      - Acepta cualquier retriever que siga el protocolo Retriever.
      - Usa GPT-4o-mini con prompt fijo para generación.
      - Reporta tokens usados para análisis de costos.
    """

    def __init__(
        self,
        retriever: Retriever,
        retriever_name: str = "dense",
        top_k: int = TOP_K,
        api_key: str = OPENAI_API_KEY,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no configurada. "
                "Agrégala en el archivo .env o como variable de entorno."
            )
        self.retriever      = retriever
        self.retriever_name = retriever_name
        self.top_k          = top_k
        self._llm           = OpenAI(api_key=api_key)

    # ── API pública ───────────────────────────────────────────────────────────

    def answer(self, question: str, k: int | None = None) -> RAGResponse:
        """
        Ejecuta el pipeline completo para una pregunta.

        Flujo:
          1. Retriever → top-k chunks como string de contexto.
          2. Construir mensajes para el LLM.
          3. Llamar GPT-4o-mini.
          4. Devolver RAGResponse con respuesta + metadata.

        Parameters
        ----------
        question : str — pregunta en lenguaje natural
        k        : int — override del top-k por defecto

        Returns
        -------
        RAGResponse
        """
        k = k or self.top_k

        # 1. Recuperar contexto
        chunks = self.retriever.retrieve(question, k)
        context = "\n\n---\n\n".join(
            c.to_context_str() if hasattr(c, "to_context_str") else str(c)
            for c in chunks
        )

        if not context.strip():
            context = "No se encontraron intervenciones relevantes para esta pregunta."

        # 2. Construir prompt
        user_content = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # 3. Llamar al LLM
        completion = self._llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

        answer_text = completion.choices[0].message.content or ""
        usage       = completion.usage

        return RAGResponse(
            question          = question,
            answer            = answer_text,
            context           = context,
            retriever_name    = self.retriever_name,
            top_k             = k,
            retrieved_chunks  = chunks,
            prompt_tokens     = usage.prompt_tokens if usage else 0,
            completion_tokens = usage.completion_tokens if usage else 0,
        )

    def batch_answer(self, questions: list[str], k: int | None = None) -> list[RAGResponse]:
        """Responde una lista de preguntas secuencialmente."""
        responses = []
        for i, q in enumerate(questions, 1):
            print(f"[RAG] Procesando pregunta {i}/{len(questions)}: {q[:60]}…")
            resp = self.answer(q, k)
            responses.append(resp)
        return responses
