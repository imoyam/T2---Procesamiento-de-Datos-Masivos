"""
parte2/graph_retriever.py  —  Parte 2.1 / 2.2
-----------------------------------------------
Recuperador GraphRAG: combina similitud vectorial (HNSW) con patrones
sobre el grafo de MillenniumDB en una sola consulta.

Expone métodos de alto nivel que corresponden a cada patrón del enunciado:
  - retrieve_by_party()         → Patrón A (restricción por partido)
  - retrieve_by_chamber()       → Patrón A (restricción por cámara)
  - retrieve_by_period()        → Patrón A (restricción por período)
  - retrieve_contrast_party()   → Patrón B (contraste por partido)
  - retrieve_contrast_chamber() → Patrón B (contraste por cámara)
  - retrieve_by_age_cohort()    → Patrón C (cohorte etaria)
  - retrieve_by_date_range()    → Patrón C (rango temporal)
  - retrieve_temporal_evol()    → Patrón C (evolución temporal)

También implementa retrieve_as_context() para que sea compatible con
RAGPipeline (mismo protocolo que DenseRetriever).
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from typing import Optional

from utils.mdb_client import MDBClient, get_client
from utils.embedder import embed_query, vec_to_list
from parte1.dense_retriever import RetrievedChunk
from parte2.query_patterns import (
    query_by_party,
    query_by_chamber,
    query_by_legislative_period,
    query_contrast_by_party,
    query_contrast_by_chamber,
    query_by_age_cohort,
    query_by_date_range,
    query_temporal_evolution,
)
from config.settings import TOP_K


def _rows_to_chunks(rows: list[dict]) -> list[RetrievedChunk]:
    """Convierte filas MDB a lista de RetrievedChunk."""
    chunks = []
    for row in rows:
        chunk = RetrievedChunk(
            chunk_id        = str(row.get("chunk_id", "")),
            intervention_id = str(row.get("intervention_id", "")),
            text            = str(row.get("text", "")),
            score           = float(row.get("score", 0.0)),
            speaker         = str(row.get("speaker", "")),
            party           = str(row.get("party", "")),
            chamber         = str(row.get("chamber", "")),
            session_date    = str(row.get("session_date", "")),
            extra           = {
                k: v for k, v in row.items()
                if k not in {"chunk_id", "intervention_id", "text",
                             "score", "speaker", "party", "chamber", "session_date"}
            },
        )
        chunks.append(chunk)
    return chunks


class GraphRetriever:
    """
    Recuperador GraphRAG sobre MillenniumDB.

    Cada método vectoriza la pregunta, construye la query MQL adecuada
    (con filtros sobre el grafo) y devuelve los chunks ordenados por score.

    La ventaja sobre el RAG denso es que los filtros estructurados se aplican
    DENTRO de la misma consulta MDB, no en post-proceso: solo se busca
    por similitud sobre el subconjunto del grafo que cumple el patrón.
    """

    def __init__(
        self,
        client: Optional[MDBClient] = None,
        top_k: int = TOP_K,
    ) -> None:
        self.client = client or get_client()
        self.top_k  = top_k

    # ── Patrón A: Restricción tipada ─────────────────────────────────────────

    def retrieve_by_party(
        self,
        question: str,
        party_name: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Top-k intervenciones similares a `question` filtradas por `party_name`.

        Por qué es mejor que el denso para esta pregunta:
            El embedding de "¿qué opina la UDI sobre X?" tiene poca señal
            sobre el partido; el denso devuelve intervenciones similares de
            cualquier partido. Aquí el WHERE pp.name = party_name garantiza
            que solo aparezcan intervenciones de ese partido.
        """
        k = k or self.top_k
        vec = vec_to_list(embed_query(question))
        query = query_by_party(vec, party_name, k)
        rows = self.client.run(query)
        return _rows_to_chunks(rows)

    def retrieve_by_chamber(
        self,
        question: str,
        chamber_name: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Top-k intervenciones similares filtradas por cámara."""
        k = k or self.top_k
        vec = vec_to_list(embed_query(question))
        query = query_by_chamber(vec, chamber_name, k)
        rows = self.client.run(query)
        return _rows_to_chunks(rows)

    def retrieve_by_period(
        self,
        question: str,
        period_id: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Top-k intervenciones similares filtradas por período legislativo."""
        k = k or self.top_k
        vec = vec_to_list(embed_query(question))
        query = query_by_legislative_period(vec, period_id, k)
        rows = self.client.run(query)
        return _rows_to_chunks(rows)

    # ── Patrón B: Agregación / contraste ─────────────────────────────────────

    def retrieve_contrast_party(
        self,
        question: str,
        k_per_party: int = 2,
        top_parties: int = 5,
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Recupera las top k_per_party intervenciones más similares POR PARTIDO.

        Devuelve un dict: { partido → [chunks] }
        Útil para construir contextos comparativos tipo:
          "Partido A dice X, Partido B dice Y"

        Estrategia:
            1. Traer los top (k_per_party * 10) resultados con partido.
            2. Agrupar en Python por partido.
            3. Tomar los top k_per_party por partido.
            4. Limitar a top_parties partidos (los que más aparecen en top).
        """
        vec = vec_to_list(embed_query(question))
        query = query_contrast_by_party(vec, k_per_party)
        rows = self.client.run(query)
        chunks = _rows_to_chunks(rows)

        # Agrupar por partido (tomando los de mayor score primero,
        # ya que vienen ordenados por score DESC)
        by_party: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            party_key = chunk.party or "Sin partido"
            if len(by_party[party_key]) < k_per_party:
                by_party[party_key].append(chunk)

        # Ordenar partidos por score máximo de su primer chunk y limitar
        sorted_parties = sorted(
            by_party.items(),
            key=lambda x: x[1][0].score if x[1] else 0,
            reverse=True,
        )
        return dict(sorted_parties[:top_parties])

    def retrieve_contrast_chamber(
        self,
        question: str,
        k_per_chamber: int = 3,
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Recupera top k_per_chamber intervenciones similares POR CÁMARA.
        Devuelve { "Senado" → [...], "Cámara de Diputadas y Diputados" → [...] }
        """
        vec = vec_to_list(embed_query(question))
        query = query_contrast_by_chamber(vec, k_per_chamber)
        rows = self.client.run(query)
        chunks = _rows_to_chunks(rows)

        by_chamber: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            cam = chunk.chamber or "Sin cámara"
            if len(by_chamber[cam]) < k_per_chamber:
                by_chamber[cam].append(chunk)

        return dict(by_chamber)

    # ── Patrón C: Atributo temporal/numérico ─────────────────────────────────

    def retrieve_by_age_cohort(
        self,
        question: str,
        birth_year_min: int,
        birth_year_max: int,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Top-k intervenciones de parlamentarios nacidos en [birth_year_min, birth_year_max].

        Ejemplo de análisis generacional:
            jóvenes  = birth_year_min=1980, birth_year_max=1999
            mayores  = birth_year_min=1940, birth_year_max=1959

        Por qué es valioso:
            Permite detectar si hay diferencias discursivas entre generaciones
            de legisladores sobre un mismo tema (ej. tecnología, medio ambiente).
        """
        k = k or self.top_k
        vec = vec_to_list(embed_query(question))
        query = query_by_age_cohort(vec, birth_year_min, birth_year_max, k)
        rows = self.client.run(query)
        return _rows_to_chunks(rows)

    def retrieve_by_date_range(
        self,
        question: str,
        date_start: str,
        date_end: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Top-k intervenciones similares dentro de un rango de fechas de sesión.

        Parameters
        ----------
        date_start, date_end : str — formato "YYYY-MM-DD"

        Por qué es mejor que el denso:
            El embedding no codifica la fecha; el denso mezcla intervenciones
            de distintos años. Este filtro permite estudiar cómo ha evolucionado
            un tema o comparar discursos antes/después de un evento político.
        """
        k = k or self.top_k
        vec = vec_to_list(embed_query(question))
        query = query_by_date_range(vec, date_start, date_end, k)
        rows = self.client.run(query)
        return _rows_to_chunks(rows)

    def retrieve_temporal_evolution(
        self,
        question: str,
        k_per_year: int = 2,
    ) -> dict[str, list[RetrievedChunk]]:
        """
        Agrupa las intervenciones más similares POR AÑO de sesión.
        Devuelve { "2019" → [...], "2020" → [...], ... }

        Útil para preguntas del tipo:
            "¿Cómo ha cambiado el discurso sobre X a lo largo del tiempo?"
        """
        vec = vec_to_list(embed_query(question))
        query = query_temporal_evolution(vec, k_per_year)
        rows = self.client.run(query)
        chunks = _rows_to_chunks(rows)

        by_year: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            raw_date = chunk.session_date or ""
            year = raw_date[:4] if len(raw_date) >= 4 else "Desconocido"
            if len(by_year[year]) < k_per_year:
                by_year[year].append(chunk)

        return dict(sorted(by_year.items()))

    # ── Interfaz compatible con RAGPipeline ──────────────────────────────────

    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        # Parámetros opcionales de contexto para el patrón adecuado:
        party: Optional[str] = None,
        chamber: Optional[str] = None,
        period: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        birth_year_min: Optional[int] = None,
        birth_year_max: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Método genérico que RAGPipeline puede llamar.
        Elige el patrón más apropiado según los parámetros presentes.
        Por defecto (sin filtros) actúa como el RAG denso.
        """
        k = k or self.top_k

        if party:
            return self.retrieve_by_party(question, party, k)
        if chamber:
            return self.retrieve_by_chamber(question, chamber, k)
        if period:
            return self.retrieve_by_period(question, period, k)
        if date_start and date_end:
            return self.retrieve_by_date_range(question, date_start, date_end, k)
        if birth_year_min is not None and birth_year_max is not None:
            return self.retrieve_by_age_cohort(question, birth_year_min, birth_year_max, k)

        # Sin filtros: equivale al denso (fallback)
        from parte1.dense_retriever import DenseRetriever
        dense = DenseRetriever(client=self.client, top_k=k)
        return dense.retrieve(question, k)

    def retrieve_as_context(
        self,
        question: str,
        k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Devuelve contexto formateado para el LLM."""
        chunks = self.retrieve(question, k, **kwargs)
        if not chunks:
            return "No se encontraron intervenciones relevantes con los filtros aplicados."
        return "\n\n---\n\n".join(c.to_context_str() for c in chunks)

    def contrast_as_context(
        self,
        question: str,
        mode: str = "party",
        **kwargs,
    ) -> str:
        """
        Devuelve contexto de contraste formateado.

        mode: "party"   → contraste por partido
              "chamber" → contraste por cámara
              "time"    → evolución temporal
        """
        if mode == "party":
            grouped = self.retrieve_contrast_party(question, **kwargs)
            sections = []
            for party, chunks in grouped.items():
                header = f"═══ {party} ═══"
                body   = "\n\n".join(c.to_context_str() for c in chunks)
                sections.append(f"{header}\n{body}")
            return "\n\n".join(sections) if sections else "Sin resultados."

        elif mode == "chamber":
            grouped = self.retrieve_contrast_chamber(question, **kwargs)
            sections = []
            for chamber, chunks in grouped.items():
                header = f"═══ {chamber} ═══"
                body   = "\n\n".join(c.to_context_str() for c in chunks)
                sections.append(f"{header}\n{body}")
            return "\n\n".join(sections) if sections else "Sin resultados."

        elif mode == "time":
            grouped = self.retrieve_temporal_evolution(question, **kwargs)
            sections = []
            for year, chunks in grouped.items():
                header = f"═══ Año {year} ═══"
                body   = "\n\n".join(c.to_context_str() for c in chunks)
                sections.append(f"{header}\n{body}")
            return "\n\n".join(sections) if sections else "Sin resultados."

        raise ValueError(f"mode='{mode}' no reconocido. Usa 'party', 'chamber' o 'time'.")
