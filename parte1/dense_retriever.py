"""
parte1/dense_retriever.py  —  Parte 1.1
-----------------------------------------
Recuperador denso puro: vectoriza la query y busca las top-k
intervenciones más similares en MillenniumDB usando el índice HNSW.

Detalles de diseño:
  - El modelo e5 es asimétrico → prefijo "query:" en la consulta.
  - Similitud coseno sobre vectores normalizados = producto punto.
  - MillenniumDB expone la función SIMILARITY() sobre el índice HNSW
    para búsqueda aproximada de vecinos.
  - Cada resultado incluye: id de intervención, texto, score de similitud
    y metadata del orador (nombre, partido, cámara, fecha de sesión).
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Optional

from utils.mdb_client import MDBClient, get_client
from utils.embedder import embed_query, vec_to_list
from config.settings import TOP_K


@dataclass
class RetrievedChunk:
    """Representa un chunk recuperado con su metadata."""
    chunk_id: str
    intervention_id: str
    text: str
    score: float
    speaker: str = ""
    party: str = ""
    chamber: str = ""         # Cámara (Senado / Diputados)
    session_date: str = ""
    legislative_period: str = ""
    extra: dict = field(default_factory=dict)

    def to_context_str(self) -> str:
        """
        Formatea el chunk como string de contexto para el LLM.
        Incluye metadata relevante como cabecera.
        """
        header_parts = []
        if self.speaker:
            header_parts.append(f"Orador: {self.speaker}")
        if self.party:
            header_parts.append(f"Partido: {self.party}")
        if self.chamber:
            header_parts.append(f"Cámara: {self.chamber}")
        if self.session_date:
            header_parts.append(f"Fecha: {self.session_date}")

        header = " | ".join(header_parts) if header_parts else "Metadata no disponible"

        return (
            f"[INT-{self.intervention_id}] ({header})\n"
            f"{self.text}\n"
            f"[Similitud: {self.score:.4f}]"
        )


class DenseRetriever:
    """
    Recuperador denso que usa MillenniumDB + índice HNSW.

    Parámetro de diseño clave: k
    ----------------------------
    k=5 es un punto de partida razonable para RAG sobre texto legislativo:
    - Suficiente contexto para el LLM sin saturar la ventana de contexto.
    - Intervenciones parlamentarias son largas → k grande puede perjudicar.
    - Se puede ajustar per-query si se necesita más cobertura temática.
    """

    # Query MQL que combina búsqueda vectorial + metadata del orador.
    # La función SIMILARITY() usa el índice HNSW (búsqueda aproximada).
    # Ajusta los nombres de etiquetas/propiedades según tu esquema MDB.
    _QUERY_TEMPLATE = """
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
          -[:RepresentsParty]->(pp :PoliticalParty)
    OPTIONAL MATCH (i)-[:IsDeliveredBy]->(pos)-[:BelongsTo]->(ch :Chamber)
    OPTIONAL MATCH (i)-[:PartOf]->(proc :Procedure)-[:TakesPlaceIn]->(s :Session)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        ch.name              AS chamber,
        s.date               AS session_date,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """

    # Query alternativa más simple si el esquema no tiene todos los nodos
    _QUERY_SIMPLE = """
    MATCH (emb :Embedding)
    RETURN
        emb.chunk_id        AS chunk_id,
        emb.intervention_id AS intervention_id,
        emb.text            AS text,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """

    def __init__(
        self,
        client: Optional[MDBClient] = None,
        top_k: int = TOP_K,
        use_metadata: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        client       : MDBClient ya conectado (o None para usar singleton global)
        top_k        : número de chunks a recuperar
        use_metadata : si True, usa la query con JOIN a metadata;
                       si False, usa la query simple (más rápida)
        """
        self.client       = client or get_client()
        self.top_k        = top_k
        self.use_metadata = use_metadata

    # ── API pública ───────────────────────────────────────────────────────────

    def retrieve(self, question: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        """
        Recupera los top-k chunks más relevantes para la pregunta.

        Flujo:
          1. Vectoriza la pregunta con e5 (prefijo "query:").
          2. Convierte el vector a lista de floats para MDB.
          3. Ejecuta la query HNSW en MDB.
          4. Parsea y devuelve los resultados como RetrievedChunk.

        Parameters
        ----------
        question : str   — pregunta en lenguaje natural
        k        : int   — override del top_k por defecto

        Returns
        -------
        list[RetrievedChunk], ordenados por score descendente
        """
        k = k or self.top_k

        # 1. Vectorizar query
        vec = embed_query(question)
        vec_list = vec_to_list(vec)

        # 2. Construir y ejecutar query MDB
        if self.use_metadata:
            query = self._QUERY_TEMPLATE.format(vec=vec_list, k=k)
        else:
            query = self._QUERY_SIMPLE.format(vec=vec_list, k=k)

        rows = self.client.run(query)

        # 3. Parsear resultados
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
            )
            chunks.append(chunk)

        return chunks

    def retrieve_as_context(self, question: str, k: Optional[int] = None) -> str:
        """
        Wrapper que devuelve el contexto ya formateado como string,
        listo para insertar en el prompt del LLM.
        """
        chunks = self.retrieve(question, k)
        if not chunks:
            return "No se encontraron intervenciones relevantes."
        return "\n\n---\n\n".join(c.to_context_str() for c in chunks)

    def retrieve_ids(self, question: str, k: Optional[int] = None) -> list[str]:
        """Devuelve solo los IDs de intervención recuperados."""
        return [c.intervention_id for c in self.retrieve(question, k)]
