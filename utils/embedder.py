"""
utils/embedder.py
-----------------
Vectorización de queries con multilingual-e5-base.

REGLAS CRÍTICAS del modelo e5 (asimétrico):
  - Documentos → prefijo "passage: "
  - Consultas  → prefijo "query: "
  - Similitud  → coseno sobre vectores L2-normalizados
                 (equivale a producto punto si los vectores están normalizados)

Los vectores DENSOS del CSV ya están precalculados con estas reglas.
Este módulo solo vectoriza QUERIES en tiempo de ejecución.
"""

from __future__ import annotations
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Instala sentence-transformers:  pip install sentence-transformers"
    )

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM, QUERY_PREFIX


class Embedder:
    """
    Singleton-friendly wrapper sobre SentenceTransformer para e5.

    Uso:
        embedder = Embedder()
        vec = embedder.embed_query("¿Qué opina la oposición sobre el litio?")
        # vec: np.ndarray de shape (768,), L2-normalizado
    """

    _instance: "Embedder | None" = None

    def __new__(cls) -> "Embedder":
        # Patrón singleton: cargamos el modelo solo una vez
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._model = None
            cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        if self._model is None:
            print(f"[Embedder] Cargando modelo '{EMBEDDING_MODEL}' …")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"[Embedder] Modelo listo (dim={EMBEDDING_DIM}).")

    # ── API pública ───────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """
        Vectoriza UNA query.

        1. Agrega el prefijo "query: " (requerido por e5).
        2. Codifica con el modelo.
        3. Normaliza L2.

        Returns
        -------
        np.ndarray shape (768,), dtype float32, normalizado.
        """
        self._load()
        prefixed = QUERY_PREFIX + text.strip()
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,   # L2-normalización interna
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """
        Vectoriza una lista de queries en batch.

        Returns
        -------
        np.ndarray shape (n, 768), cada fila normalizada.
        """
        self._load()
        prefixed = [QUERY_PREFIX + t.strip() for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        return vecs.astype(np.float32)

    def vec_to_mdb_list(self, vec: np.ndarray) -> list[float]:
        """
        Convierte un vector numpy a lista Python de floats,
        formato que espera MillenniumDB en las consultas de similitud.
        """
        return vec.tolist()


# ── Función de conveniencia ───────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    """Shortcut: Embedder().embed_query(text)."""
    return Embedder().embed_query(text)


def vec_to_list(vec: np.ndarray) -> list[float]:
    """Shortcut: convierte ndarray → list[float] para MDB."""
    return vec.tolist()
