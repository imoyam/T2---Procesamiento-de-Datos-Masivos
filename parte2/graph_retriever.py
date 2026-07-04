from __future__ import annotations
import sys, os, csv
from io import StringIO
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
    """Convierte filas aplastadas de MDB a lista de RetrievedChunk."""
    chunks = []
    for row in rows:
        if not row: continue
        
        # El cliente HTTP aplasta las claves y valores en un solo string
        raw_keys = list(row.keys())[0]
        raw_vals = list(row.values())[0]
        
        # Mapeamos dinámicamente y le QUITAMOS el '?' a las llaves que devuelve MDB
        keys = [k.strip().replace('?', '') for k in raw_keys.split(',')]
        try:
            vals = next(csv.reader(StringIO(raw_vals)))
            data = dict(zip(keys, vals))
        except Exception:
            continue
            
        content = data.get("emb.content", "")
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
            
        score_str = data.get("dist", "null")
        
        chunk = RetrievedChunk(
            chunk_id        = data.get("emb.chunk_id", ""),
            intervention_id = data.get("i.id", ""),
            text            = content,
            score           = float(score_str) if score_str != "null" else 999.0,
            speaker         = data.get("pos.name", ""),
            party           = data.get("pp.name", ""),
            chamber         = data.get("ch.name", ""),
            session_date    = data.get("s.date", "")
        )
        chunks.append(chunk)
    return chunks


class GraphRetriever:
    def __init__(self, client: Optional[MDBClient] = None, top_k: int = TOP_K) -> None:
        self.client = client or get_client()
        self.top_k  = top_k

    def retrieve_by_party(self, question: str, party_name: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_by_party(vec_str, party_name, k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_by_chamber(self, question: str, chamber_name: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_by_chamber(vec_str, chamber_name, k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_by_period(self, question: str, period_id: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_by_legislative_period(vec_str, period_id, k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_contrast_party(self, question: str, k_per_party: int = 2, top_parties: int = 5) -> dict[str, list[RetrievedChunk]]:
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_contrast_by_party(vec_str, k_per_party)
        chunks = _rows_to_chunks(self.client.run(query))

        by_party: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            party_key = chunk.party or "Sin partido"
            if len(by_party[party_key]) < k_per_party:
                by_party[party_key].append(chunk)

        sorted_parties = sorted(by_party.items(), key=lambda x: x[1][0].score if x[1] else 999.0)
        return dict(sorted_parties[:top_parties])

    def retrieve_contrast_chamber(self, question: str, k_per_chamber: int = 3) -> dict[str, list[RetrievedChunk]]:
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_contrast_by_chamber(vec_str, k_per_chamber)
        chunks = _rows_to_chunks(self.client.run(query))

        by_chamber: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            cam = chunk.chamber or "Sin cámara"
            if len(by_chamber[cam]) < k_per_chamber:
                by_chamber[cam].append(chunk)
        return dict(by_chamber)

    def retrieve_by_age_cohort(self, question: str, birth_year_min: int, birth_year_max: int, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_by_age_cohort(vec_str, birth_year_min, birth_year_max, k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_by_date_range(self, question: str, date_start: str, date_end: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_by_date_range(vec_str, date_start, date_end, k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_temporal_evolution(self, question: str, k_per_year: int = 2) -> dict[str, list[RetrievedChunk]]:
        vec_str = str(vec_to_list(embed_query(question)))
        query = query_temporal_evolution(vec_str, k_per_year)
        chunks = _rows_to_chunks(self.client.run(query))

        by_year: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            raw_date = chunk.session_date or ""
            year = raw_date[:4] if len(raw_date) >= 4 else "Desconocido"
            if len(by_year[year]) < k_per_year:
                by_year[year].append(chunk)
        return dict(sorted(by_year.items()))

    def retrieve(self, question: str, k: Optional[int] = None, party: Optional[str] = None, chamber: Optional[str] = None, period: Optional[str] = None, date_start: Optional[str] = None, date_end: Optional[str] = None, birth_year_min: Optional[int] = None, birth_year_max: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        if party: return self.retrieve_by_party(question, party, k)
        if chamber: return self.retrieve_by_chamber(question, chamber, k)
        if period: return self.retrieve_by_period(question, period, k)
        if date_start and date_end: return self.retrieve_by_date_range(question, date_start, date_end, k)
        if birth_year_min is not None and birth_year_max is not None: return self.retrieve_by_age_cohort(question, birth_year_min, birth_year_max, k)

        from parte1.dense_retriever import DenseRetriever
        dense = DenseRetriever(client=self.client, top_k=k)
        return dense.retrieve(question, k)

    def retrieve_as_context(self, question: str, k: Optional[int] = None, **kwargs) -> str:
        chunks = self.retrieve(question, k, **kwargs)
        if not chunks: return "No se encontraron intervenciones relevantes."
        return "\n\n---\n\n".join(c.to_context_str() for c in chunks)

    def contrast_as_context(self, question: str, mode: str = "party", **kwargs) -> str:
        if mode == "party":
            grouped = self.retrieve_contrast_party(question, **kwargs)
        elif mode == "chamber":
            grouped = self.retrieve_contrast_chamber(question, **kwargs)
        elif mode == "time":
            grouped = self.retrieve_temporal_evolution(question, **kwargs)
        else:
            raise ValueError(f"mode='{mode}' no reconocido.")
            
        sections = []
        for key, chunks in grouped.items():
            header = f"═══ {key} ═══"
            body   = "\n\n".join(c.to_context_str() for c in chunks)
            sections.append(f"{header}\n{body}")
        return "\n\n".join(sections) if sections else "Sin resultados."