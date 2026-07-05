# Parte 3 - Recuperacion hibrida

Esta carpeta implementa la Parte 3 del enunciado: un recuperador sparse propio y un recuperador hibrido dense+sparse.

## Diseno

- Sparse: BM25 implementado en Python puro, sin motor externo. Es adecuado para terminos exactos, siglas como CAE/FES, apellidos, nombres de leyes y numeros de boletin.
- Store: `part3/sparse_store/bm25_index.pkl.gz`. El store guarda documentos, postings, largos e IDs. Se ignora por Git porque deriva de los datos y no debe subirse.
- Llaves: cada documento usa `chunk_id` como `doc_id` primario y mantiene `intervention_id` para citar y cruzar con MillenniumDB.
- RAG sparse: `SparseRetriever` expone `retrieve` y `retrieve_as_context`, igual que el denso.
- Fusion + rerank: `HybridRetriever` recupera pools densos y sparse, fusiona con Reciprocal Rank Fusion y reordena con un criterio deterministico tipo MMR/Jaccard. No usa LLM para rerank.

## Uso

Construir el indice:

```bash
python part3/run_parte3.py --build-index --build-only
```

Ejecutar recuperacion sparse e hibrida sin llamar al LLM:

```bash
python part3/run_parte3.py --no-llm
```

Ejecutar el pipeline RAG completo con `gpt-4o-mini`:

```bash
python part3/run_parte3.py
```

Para pruebas rapidas se puede limitar el numero de documentos indexados:

```bash
python part3/run_parte3.py --force-rebuild --limit 1000 --no-llm
```

Los resultados quedan en `part3/resultados_parte3_YYYYMMDD_HHMMSS.json`.
