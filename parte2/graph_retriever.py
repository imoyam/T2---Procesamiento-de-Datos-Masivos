from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from io import StringIO
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TOP_K
from parte1.dense_retriever import RetrievedChunk
from parte2.query_patterns import (
    query_by_age_cohort,
    query_by_chamber,
    query_by_date_range,
    query_by_legislative_period,
    query_by_party,
    query_contrast_by_chamber,
    query_contrast_by_party,
    query_temporal_evolution,
)
from utils.embedder import embed_query, vec_to_list
from utils.mdb_client import MDBClient, get_client


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "null":
        return ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text


def _normalize_row(row: dict) -> dict[str, str]:
    if not row:
        return {}

    if len(row) == 1:
        raw_keys = next(iter(row.keys()))
        raw_vals = next(iter(row.values()))
        keys = [key.strip().replace("?", "") for key in raw_keys.split(",")]
        try:
            vals = next(csv.reader(StringIO(str(raw_vals))))
        except Exception:
            vals = [raw_vals]
        return {key: _clean(value) for key, value in zip(keys, vals)}

    return {key.strip().replace("?", ""): _clean(value) for key, value in row.items()}


def _pick(data: dict[str, str], *names: str) -> str:
    for name in names:
        key = name.replace("?", "")
        if data.get(key):
            return data[key]
    return ""


def _score(data: dict[str, str]) -> float:
    raw = _pick(data, "dist")
    try:
        return float(raw) if raw else 999.0
    except ValueError:
        return 999.0


def _rows_to_chunks(rows: list[dict]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for row in rows:
        data = _normalize_row(row)
        if not data:
            continue

        intervention_id = _pick(data, "i", "i.id", "intervention_id")
        chunk_id = _pick(data, "emb", "emb.chunk_id", "chunk_id")
        text = _pick(data, "emb.content", "content", "i.transcription")
        speaker = _pick(data, "person.full_name", "pos.name", "speaker")
        party = _pick(data, "pp.name", "party")
        chamber = _pick(data, "pos.role", "ch.name", "chamber")
        session_date = _pick(data, "s.date", "session_date")
        birth_date = _pick(data, "person.birth_date", "birth_date")
        period = _pick(data, "lp.name", "period")

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                intervention_id=intervention_id,
                text=text,
                score=_score(data),
                speaker=speaker,
                party=party,
                chamber=chamber,
                session_date=session_date,
                extra={
                    "birth_date": birth_date,
                    "legislative_period": period,
                },
            )
        )
    return chunks


class GraphRetriever:
    def __init__(self, client: Optional[MDBClient] = None, top_k: int = TOP_K) -> None:
        self.client = client or get_client()
        self.top_k = top_k

    def _vector(self, question: str) -> str:
        return str(vec_to_list(embed_query(question)))

    def retrieve_by_party(self, question: str, party_name: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        query = query_by_party(self._vector(question), party_name, k or self.top_k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_by_chamber(self, question: str, chamber_name: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        query = query_by_chamber(self._vector(question), chamber_name, k or self.top_k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_by_period(self, question: str, period_name: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        query = query_by_legislative_period(self._vector(question), period_name, k or self.top_k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_contrast_party(
        self,
        question: str,
        k_per_party: int = 2,
        top_parties: int = 5,
    ) -> dict[str, list[RetrievedChunk]]:
        query = query_contrast_by_party(self._vector(question), k_per_party, top_parties)
        chunks = _rows_to_chunks(self.client.run(query))

        by_party: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            key = chunk.party or "Sin partido"
            if len(by_party[key]) < k_per_party:
                by_party[key].append(chunk)

        sorted_parties = sorted(by_party.items(), key=lambda item: item[1][0].score if item[1] else 999.0)
        return dict(sorted_parties[:top_parties])

    def retrieve_contrast_chamber(self, question: str, k_per_chamber: int = 3) -> dict[str, list[RetrievedChunk]]:
        query = query_contrast_by_chamber(self._vector(question), k_per_chamber)
        chunks = _rows_to_chunks(self.client.run(query))

        by_chamber: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            key = chunk.chamber or "Sin camara"
            if len(by_chamber[key]) < k_per_chamber:
                by_chamber[key].append(chunk)
        return dict(by_chamber)

    def retrieve_by_age_cohort(
        self,
        question: str,
        birth_year_min: int,
        birth_year_max: int,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        query = query_by_age_cohort(self._vector(question), birth_year_min, birth_year_max, k or self.top_k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_by_date_range(
        self,
        question: str,
        date_start: str,
        date_end: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        query = query_by_date_range(self._vector(question), date_start, date_end, k or self.top_k)
        return _rows_to_chunks(self.client.run(query))

    def retrieve_temporal_evolution(self, question: str, k_per_year: int = 2) -> dict[str, list[RetrievedChunk]]:
        query = query_temporal_evolution(self._vector(question), k_per_year)
        chunks = _rows_to_chunks(self.client.run(query))

        by_year: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            year = chunk.session_date[:4] if len(chunk.session_date) >= 4 else "Desconocido"
            if len(by_year[year]) < k_per_year:
                by_year[year].append(chunk)
        return dict(sorted(by_year.items()))

    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        party: Optional[str] = None,
        chamber: Optional[str] = None,
        period: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        birth_year_min: Optional[int] = None,
        birth_year_max: Optional[int] = None,
    ) -> list[RetrievedChunk]:
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

        from parte1.dense_retriever import DenseRetriever

        dense = DenseRetriever(client=self.client, top_k=k)
        return dense.retrieve(question, k)

    def retrieve_as_context(self, question: str, k: Optional[int] = None, **kwargs) -> str:
        chunks = self.retrieve(question, k, **kwargs)
        if not chunks:
            return "No se encontraron intervenciones relevantes."
        return "\n\n---\n\n".join(chunk.to_context_str() for chunk in chunks)

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
            body = "\n\n".join(chunk.to_context_str() for chunk in chunks)
            sections.append(f"=== {key} ===\n{body}")
        return "\n\n".join(sections) if sections else "Sin resultados."
