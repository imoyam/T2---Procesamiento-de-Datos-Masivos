"""
config/settings.py
------------------
Configuracion central del proyecto. Lee variables desde .env o entorno.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# MillenniumDB
MDB_HOST = os.getenv("MDB_HOST", "localhost")
MDB_PORT = int(os.getenv("MDB_PORT", "8080"))
MDB_DB_NAME = os.getenv("MDB_DB_NAME", "legislativo")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Modelo de embeddings
# multilingual-e5-base: modelo asimetrico, dim=768.
# Prefijos obligatorios: "passage:" para documentos, "query:" para consultas.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768
DOC_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# LLM
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.2

# RAG
TOP_K = int(os.getenv("TOP_K", "5"))

# Prompt del sistema (fijo en todas las partes)
SYSTEM_PROMPT = """Eres un asistente especializado en el poder legislativo chileno.
Tu tarea es responder preguntas usando UNICAMENTE el contexto de intervenciones parlamentarias
que se te entrega. Sigue estas reglas:

1. Responde en espanol, de manera clara y estructurada.
2. Cita siempre la fuente de cada afirmacion indicando el ID de intervencion entre corchetes,
   por ejemplo: [INT-i12345].
3. Si el contexto no contiene informacion suficiente para responder, dilo explicitamente.
4. No inventes datos ni extrapoles mas alla del contexto proporcionado.
5. Si hay posturas distintas entre parlamentarios o partidos, presentalas de forma equilibrada.
"""

USER_PROMPT_TEMPLATE = """Contexto de intervenciones parlamentarias:
{context}

---
Pregunta: {question}

Responde basandote unicamente en las intervenciones anteriores y cita sus IDs."""
