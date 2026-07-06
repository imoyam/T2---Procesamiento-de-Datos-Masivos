# RAG Legislativo Chileno

Sistema de preguntas y respuestas sobre intervenciones del poder legislativo chileno.

## Estructura del Proyecto

```
rag_legislativo/
├── config/
│   └── settings.py          # Configuración central (claves, parámetros)
├── utils/
│   ├── mdb_client.py        # Cliente MillenniumDB reutilizable
│   └── embedder.py          # Vectorización de queries con multilingual-e5-base
├── parte1/
│   ├── dense_retriever.py   # Recuperador denso (similitud coseno sobre HNSW)
│   ├── rag_pipeline.py      # Pipeline RAG completo (retriever + GPT-4o-mini)
│   └── run_parte1.py        # Script de ejecución y pruebas Parte 1
├── parte2/
│   ├── graph_retriever.py   # Recuperador GraphRAG con patrones sobre grafo
│   ├── query_patterns.py    # Patrones de consulta MDB (partido, cámara, temporal)
│   └── run_parte2.py        # Script de ejecución y comparación Parte 2
└── tests/
    └── hello_world.py       # Parte 0.3: validación de carga MDB
```

## Requisitos

```bash
pip install millenniumdb-driver sentence-transformers openai numpy pandas python-dotenv
```

## Configuración

Copiar `.env.example` a `.env` y completar las claves:
```
MDB_HOST=localhost
MDB_PORT=8080
OPENAI_API_KEY=<tu_clave>
```

## Ejecución

### Parte 0 — Validación de carga
```bash
python tests/hello_world.py
```

### Parte 1 — RAG Denso
```bash
python parte1/run_parte1.py
```

### Parte 2 — GraphRAG
```bash
python parte2/run_parte2.py
```

