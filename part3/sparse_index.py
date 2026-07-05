from __future__ import annotations
import csv
import gzip
import math
import pickle
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "sparse_store" / "bm25_index.pkl.gz"

SPANISH_STOPWORDS = {
    "a", "al", "algo", "ante", "antes", "como", "con", "contra", "cual",
    "cuando", "de", "del", "desde", "donde", "dos", "e", "el", "ella",
    "ellos", "en", "entre", "era", "es", "esa", "ese", "eso", "esta",
    "este", "esto", "fue", "ha", "han", "hay", "la", "las", "le", "les",
    "lo", "los", "mas", "me", "mi", "mientras", "muy", "no", "nos", "o",
    "para", "pero", "por", "que", "quien", "se", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tambien", "te", "tiene", "un", "una",
    "unas", "uno", "unos", "y", "ya",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

@dataclass
class SparseDocument:
    doc_id: str
    intervention_id: str
    chunk_id: str
    text: str
    speaker: str = ""
    party: str = ""
    chamber: str = ""
    session_date: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class SparseSearchResult:
    document: SparseDocument
    score: float
    rank: int
    matched_terms: list[str]

def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in SPANISH_STOPWORDS and len(tok) > 1]
    return tokens

def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "null":
        return ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.replace('\\"', '"')

def _parse_mdb_row(row: dict[str, Any]) -> dict[str, str]:
    if not row:
        return {}
    raw_keys = list(row.keys())[0]
    raw_vals = list(row.values())[0]
    keys = [k.strip().replace("?", "") for k in raw_keys.split(",")]
    try:
        vals = next(csv.reader(StringIO(str(raw_vals))))
        data = dict(zip(keys, vals))
    except Exception:
        return {}
    return {k: _clean_value(v) for k, v in data.items()}

def rows_to_documents(rows: Iterable[dict[str, Any]]) -> list[SparseDocument]:
    docs: list[SparseDocument] = []
    seen: set[str] = set()

    for row in rows:
        data = _parse_mdb_row(row)
        if not data:
            continue

        chunk_id = data.get("emb.chunk_id", "")
        intervention_id = data.get("i.id", "")
        text = data.get("emb.content", "")

        doc_id = chunk_id or intervention_id or uuid.uuid4().hex
        
        if not text or doc_id in seen:
            continue

        seen.add(doc_id)
        docs.append(
            SparseDocument(
                doc_id=doc_id,
                intervention_id=intervention_id,
                chunk_id=chunk_id,
                text=text,
            )
        )

    return docs

class BM25Index:
    def __init__(
        self,
        documents: Optional[list[SparseDocument]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[SparseDocument] = documents or []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.id_to_index: dict[str, int] = {}

        if documents:
            self.build(documents)

    @classmethod
    def from_mdb(cls, client: Any, limit: Optional[int] = None) -> "BM25Index":
        # Usamos LIMIT 3000 para que indexe en 5 segundos y evite congelarse
        # Y solo traemos lo vital para no sobrecargar el motor
        limite_real = limit if limit else 3000
        query = f"""
        MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding)
        RETURN ?emb.chunk_id, ?i.id, ?emb.content
        LIMIT {limite_real}
        """
        rows = client.run(query)
        docs = rows_to_documents(rows)
        return cls(docs)

    def build(self, documents: list[SparseDocument]) -> None:
        self.documents = documents
        self.doc_lengths = []
        self.postings = {}
        self.id_to_index = {}

        for doc_idx, doc in enumerate(documents):
            self.id_to_index[doc.doc_id] = doc_idx
            if doc.chunk_id:
                self.id_to_index[doc.chunk_id] = doc_idx
            if doc.intervention_id:
                self.id_to_index.setdefault(doc.intervention_id, doc_idx)

            counts: dict[str, int] = {}
            for token in tokenize(doc.text):
                counts[token] = counts.get(token, 0) + 1
            self.doc_lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self.postings.setdefault(token, []).append((doc_idx, tf))

        total_length = sum(self.doc_lengths)
        self.avg_doc_length = total_length / len(self.doc_lengths) if self.doc_lengths else 0.0

    @property
    def size(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_PATH) -> "BM25Index":
        with gzip.open(Path(path), "rb") as fh:
            return pickle.load(fh)

    def get_document(self, doc_id: str) -> Optional[SparseDocument]:
        idx = self.id_to_index.get(doc_id)
        if idx is None:
            return None
        return self.documents[idx]

    def search(self, query: str, top_k: int = 5, candidate_ids: Optional[set[str]] = None) -> list[SparseSearchResult]:
        if not self.documents:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        n_docs = len(self.documents)
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        allowed_indexes: Optional[set[int]] = None

        if candidate_ids is not None:
            allowed_indexes = {
                idx for doc_id in candidate_ids
                if (idx := self.id_to_index.get(doc_id)) is not None
            }

        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                if allowed_indexes is not None and doc_idx not in allowed_indexes:
                    continue

                doc_len = self.doc_lengths[doc_idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avg_doc_length or 1.0))
                contribution = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution
                matched.setdefault(doc_idx, set()).add(term)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SparseSearchResult(
                document=self.documents[doc_idx],
                score=score,
                rank=rank,
                matched_terms=sorted(matched.get(doc_idx, set())),
            )
            for rank, (doc_idx, score) in enumerate(ranked, start=1)
        ]