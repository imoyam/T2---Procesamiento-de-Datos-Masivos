from __future__ import annotations


def _q(value: str) -> str:
    """Escape a Python string for a MillenniumDB quoted literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _role_from_chamber(chamber_name: str) -> str:
    normalized = chamber_name.strip().lower()
    if normalized in {"senado", "senador", "senadores"}:
        return "Senador"
    if normalized in {"camara", "camara de diputados", "c.diputados", "diputados", "diputado"}:
        return "Diputado"
    return chamber_name


def query_by_party(vec_str: str, party_name: str, k: int) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty {{name: "{_q(party_name)}"}}),
          (?person :Person)-[:ServedAs]->(?pos)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?dist
    LIMIT {k}
    """


def query_by_chamber(vec_str: str, chamber_name: str, k: int) -> str:
    role = _role_from_chamber(chamber_name)
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?person :Person)-[:ServedAs]->(?pos)
    WHERE ?pos.role = "{_q(role)}"
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?dist
    LIMIT {k}
    """


def query_by_legislative_period(vec_str: str, period_name: str, k: int) -> str:
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?person :Person)-[:ServedAs]->(?pos),
          (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)-[:BelongsTo]->(?leg :Legislature)-[:BelongsTo]->(?lp :LegislativePeriod {{name: "{_q(period_name)}"}})
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?s.date, ?lp.name, ?dist
    LIMIT {k}
    """


def query_contrast_by_party(vec_str: str, k_per_party: int, top_parties: int = 5) -> str:
    k_total = k_per_party * max(top_parties * 4, 10)
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?person :Person)-[:ServedAs]->(?pos)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?dist
    LIMIT {k_total}
    """


def query_contrast_by_chamber(vec_str: str, k_per_chamber: int) -> str:
    k_total = k_per_chamber * 8
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?person :Person)-[:ServedAs]->(?pos)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?dist
    LIMIT {k_total}
    """


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
    WHERE ?person.birth_date >= "{birth_year_min:04d}-01-01T00:00:00"
      AND ?person.birth_date <= "{birth_year_max:04d}-12-31T23:59:59"
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?person.birth_date, ?dist
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
          (?person :Person)-[:ServedAs]->(?pos),
          (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)
    WHERE ?s.date >= "{_q(date_start)}"
      AND ?s.date <= "{_q(date_end)}"
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?s.date, ?dist
    LIMIT {k}
    """


def query_temporal_evolution(vec_str: str, k_per_year: int) -> str:
    k_total = k_per_year * 20
    return f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
          (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
          (?person :Person)-[:ServedAs]->(?pos),
          (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)
    LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vec_str}"))
    ORDER BY ?dist
    RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?s.date, ?dist
    LIMIT {k_total}
    """
