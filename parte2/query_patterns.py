from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN A: Restricción tipada (partido / cámara / período)
# ─────────────────────────────────────────────────────────────────────────────

def query_by_party(vec_str: str, party_name: str, k: int) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty {{name: "{party_name}"}})
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?dist
    LIMIT {k}
    """

def query_by_chamber(vec_str: str, chamber_name: str, k: int) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:BelongsTo]->(?ch :Chamber {{name: "{chamber_name}"}}),
          (?pos)-[:Represents]->(?pp :PoliticalParty)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?ch.name, ?dist
    LIMIT {k}
    """

def query_by_legislative_period(vec_str: str, period_id: str, k: int) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)-[:BelongsTo]->(?leg :Legislature)-[:BelongsTo]->(?lp :LegislativePeriod {{id: "{period_id}"}})
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?s.date, ?dist
    LIMIT {k}
    """

# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN B: Agregación / contraste entre partidos o cámaras
# ─────────────────────────────────────────────────────────────────────────────

def query_contrast_by_party(vec_str: str, k_per_party: int) -> str:
    k_total = k_per_party * 10 
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?dist
    LIMIT {k_total}
    """

def query_contrast_by_chamber(vec_str: str, k: int) -> str:
    k_total = k * 4
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:BelongsTo]->(?ch :Chamber),
          (?pos)-[:Represents]->(?pp :PoliticalParty)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?ch.name, ?dist
    LIMIT {k_total}
    """

# ─────────────────────────────────────────────────────────────────────────────
# PATRÓN C: Atributo numérico/temporal
# ─────────────────────────────────────────────────────────────────────────────

def query_by_age_cohort(
    vec_str: str,
    birth_year_min: int,
    birth_year_max: int,
    k: int,
) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?person :Person)-[:ServedAs]->(?pos)
    WHERE ?person.birth_year >= {birth_year_min}
      AND ?person.birth_year <= {birth_year_max}
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?person.birth_year, ?dist
    LIMIT {k}
    """

def query_by_date_range(
    vec_str: str,
    date_start: str,
    date_end: str,
    k: int,
) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)
    WHERE ?s.date >= "{date_start}"
      AND ?s.date <= "{date_end}"
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?s.date, ?dist
    LIMIT {k}
    """

def query_temporal_evolution(vec_str: str, k_per_year: int) -> str:
    k_total = k_per_year * 10
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb.chunk_id, ?i.id, ?emb.content, ?pos.name, ?pp.name, ?s.date, ?dist
    LIMIT {k_total}
    """