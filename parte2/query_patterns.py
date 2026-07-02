"""
parte2/query_patterns.py  —  Parte 2.1
-----------------------------------------
Patrones de consulta MQL para GraphRAG en MillenniumDB.

Implementa los tres patrones requeridos por el enunciado:

  PATRÓN A — Recuperación con restricción tipada
      Filtra por partido, cámara o período legislativo antes/después
      de la búsqueda semántica. Ejemplo: "¿qué dice el partido X sobre Y?"

  PATRÓN B — Agregación / contraste
      Recupera chunks similares y los agrupa por partido o cámara.
      Permite comparar posturas entre actores políticos.

  PATRÓN C — Atributo numérico/temporal
      Filtra o agrupa por cohorte etaria (fecha de nacimiento) o
      por fecha de intervención. Permite estudiar cómo ha evolucionado
      un tema en el tiempo o si hay diferencias generacionales.

Cada patrón devuelve una query MQL (string) que MillenniumDB ejecutará
combinando el índice HNSW con traversals sobre el grafo.

POR QUÉ ESTO ES MEJOR QUE EL RAG DENSO:
  El RAG denso recupera por similitud semántica global: no distingue
  de qué partido viene una intervención ni en qué fecha. Si preguntas
  "¿qué opina la UDI sobre las pensiones?", el denso puede traer
  intervenciones de cualquier partido con contenido similar. El grafo
  añade un filtro estructurado que el embedding no puede proveer.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN A: Restricción tipada (partido / cámara / período)
# ─────────────────────────────────────────────────────────────────────────────

def query_by_party(vec: list[float], party_name: str, k: int) -> str:
    """
    Recupera las top-k intervenciones semánticamente similares al vector,
    FILTRADAS por partido político.

    Ejemplo de uso:
        "¿Qué opina el Partido Socialista sobre la reforma de pensiones?"
        → party_name = "Partido Socialista"

    Por qué es mejor que el denso:
        Sin este filtro, el denso devuelve las intervenciones más similares
        sin importar el partido; aquí garantizamos que el contexto
        corresponde al actor político preguntado.
    """
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
          -[:RepresentsParty]->(pp :PoliticalParty)
    WHERE pp.name = "{party_name}"
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """


def query_by_chamber(vec: list[float], chamber_name: str, k: int) -> str:
    """
    Filtra por cámara: "Senado" o "Cámara de Diputadas y Diputados".

    Ejemplo:
        "¿Qué ha discutido el Senado sobre el litio?"
    """
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)-[:BelongsTo]->(ch :Chamber)
    WHERE ch.name = "{chamber_name}"
    OPTIONAL MATCH (pos)-[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        ch.name              AS chamber,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """


def query_by_legislative_period(vec: list[float], period_id: str, k: int) -> str:
    """
    Filtra por período legislativo (ej. "355" o el ID que use MDB).

    Ejemplo:
        "¿Qué se dijo sobre seguridad en el período legislativo actual?"
    """
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
    OPTIONAL MATCH (i)-[:PartOf]->(proc :Procedure)
                   -[:TakesPlaceIn]->(s :Session)
                   -[:PartOf]->(leg :Legislature)
                   -[:PartOf]->(lp :LegislativePeriod)
    WHERE lp.id = "{period_id}"
    OPTIONAL MATCH (pos)-[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        s.date               AS session_date,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """


# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN B: Agregación / contraste entre partidos o cámaras
# ─────────────────────────────────────────────────────────────────────────────

def query_contrast_by_party(vec: list[float], k_per_party: int) -> str:
    """
    Recupera las top-k_per_party intervenciones más similares POR CADA PARTIDO.

    Esto permite contrastar cómo distintos partidos hablan de un mismo tema.
    MillenniumDB no tiene PARTITION BY, pero podemos hacer un ORDER BY
    y post-procesar en Python, o usar subconsultas.

    Estrategia: traer los top k*N resultados y filtrar en Python por partido.
    Aquí traemos los top (k_per_party * 10) y el GraphRetriever agrupa.

    Ejemplo:
        "Contrasta las posturas de distintos partidos sobre la migración"
    """
    k_total = k_per_party * 10  # margen para cubrir varios partidos
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
          -[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k_total}
    """


def query_contrast_by_chamber(vec: list[float], k: int) -> str:
    """
    Recupera intervenciones similares agrupables por cámara.
    Útil para comparar cómo Senado vs Diputados tratan un tema.
    """
    k_total = k * 4
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)-[:BelongsTo]->(ch :Chamber)
    OPTIONAL MATCH (pos)-[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        ch.name              AS chamber,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k_total}
    """


# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN C: Atributo numérico/temporal
# ─────────────────────────────────────────────────────────────────────────────

def query_by_age_cohort(
    vec: list[float],
    birth_year_min: int,
    birth_year_max: int,
    k: int,
) -> str:
    """
    Filtra intervenciones de parlamentarios nacidos entre birth_year_min y max.
    Permite analizar si hay diferencias generacionales en cómo se habla de un tema.

    Ejemplo:
        "¿Los parlamentarios más jóvenes (nacidos después de 1980) hablan
         distinto sobre tecnología y digitalización?"
        → birth_year_min=1980, birth_year_max=2000

    Por qué es valioso:
        La similitud semántica no captura la edad del orador.
        Con el grafo podemos segmentar la recuperación por cohorte generacional
        y comparar los discursos.
    """
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
    MATCH (person :Person)-[:SavedAs]->(pos)
    WHERE person.birth_year >= {birth_year_min}
      AND person.birth_year <= {birth_year_max}
    OPTIONAL MATCH (pos)-[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id          AS chunk_id,
        i.id                  AS intervention_id,
        emb.text              AS text,
        pos.name              AS speaker,
        pp.name               AS party,
        person.birth_year     AS birth_year,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """


def query_by_date_range(
    vec: list[float],
    date_start: str,
    date_end: str,
    k: int,
) -> str:
    """
    Filtra intervenciones según la fecha de la sesión.
    Permite estudiar cómo ha evolucionado un tema en el tiempo.

    Parameters
    ----------
    date_start : str  — fecha inicio, formato "YYYY-MM-DD"
    date_end   : str  — fecha fin,   formato "YYYY-MM-DD"

    Ejemplo:
        "¿Cómo ha cambiado el discurso sobre la educación entre 2018 y 2022?"
    """
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
    OPTIONAL MATCH (i)-[:PartOf]->(proc :Procedure)-[:TakesPlaceIn]->(s :Session)
    WHERE s.date >= "{date_start}"
      AND s.date <= "{date_end}"
    OPTIONAL MATCH (pos)-[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        s.date               AS session_date,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k}
    """


def query_temporal_evolution(vec: list[float], k_per_period: int) -> str:
    """
    Recupera intervenciones sobre un tema agrupables por año/período,
    para analizar la evolución temporal del discurso legislativo.

    Devuelve hasta k_per_period * 10 resultados con fecha incluida.
    El GraphRetriever agrupa y selecciona por año en Python.
    """
    k_total = k_per_period * 10
    return f"""
    MATCH (emb :Embedding)-[:EmbeddingOf]->(i :Intervention)
          -[:IsDeliveredBy]->(pos :Position)
    OPTIONAL MATCH (i)-[:PartOf]->(proc :Procedure)-[:TakesPlaceIn]->(s :Session)
    OPTIONAL MATCH (pos)-[:RepresentsParty]->(pp :PoliticalParty)
    RETURN
        emb.chunk_id         AS chunk_id,
        i.id                 AS intervention_id,
        emb.text             AS text,
        pos.name             AS speaker,
        pp.name              AS party,
        s.date               AS session_date,
        SIMILARITY(emb.vector, {vec}) AS score
    ORDER BY score DESC
    LIMIT {k_total}
    """
