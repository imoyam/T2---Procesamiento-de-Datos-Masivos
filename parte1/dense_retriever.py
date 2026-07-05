import sys, os, csv
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from utils.mdb_client import MDBClient
from utils.embedder import embed_query, vec_to_list
from config.settings import TOP_K

@dataclass
class RetrievedChunk:
    intervention_id: str
    text: str
    score: float
    chunk_id: str = ""
    speaker: str = ""
    party: str = ""
    chamber: str = ""
    session_date: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_context_str(self) -> str:
        party_str = f" ({self.party})" if self.party else ""
        return f"[INT-{self.intervention_id}]{party_str} {self.text}"

class DenseRetriever:
    def __init__(self, client: MDBClient = None, top_k: int = TOP_K, use_metadata: bool = False) -> None:
        self.client = client or MDBClient()
        self.top_k = top_k
        self.use_metadata = use_metadata

    def retrieve(self, question: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        
        # 1. Convertimos el vector a un string que incluye los corchetes
        vector_str = str(vec_to_list(embed_query(question)))
        
        # 2. La query definitiva con el casteo explícito tensorFloat()
        query = f"""
        MATCH (?i :Intervention)-[:HasEmbedding]->(?emb :Embedding),
              (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?pp :PoliticalParty),
              (?person :Person)-[:ServedAs]->(?pos)
        LET ?dist = COSINE_DISTANCE(?emb.value, tensorFloat("{vector_str}"))
        ORDER BY ?dist
        RETURN ?emb, ?i, ?emb.content, ?person.full_name, ?pp.name, ?pos.role, ?dist
        LIMIT {k}
        """
        filas = self.client.run(query)
        chunks = []
        
        for f in filas:
            if not f: continue
            
            try:
                vals = list(f.values())
                
                # Manejador robusto: Si la BD lo devuelve separado en múltiples columnas
                if len(vals) >= 7:
                    c_id = str(vals[0])
                    i_id = str(vals[1])
                    texto = str(vals[2])
                    speaker = str(vals[3])
                    party = str(vals[4])
                    chamber = str(vals[5])
                    dist_str = str(vals[6])
                    
                # Manejador robusto: Si la BD lo devuelve aplastado en 1 columna (separado por comas)
                elif len(vals) == 1:
                    parsed = next(csv.reader(StringIO(vals[0])))
                    c_id = str(parsed[0])
                    i_id = str(parsed[1])
                    texto = str(parsed[2])
                    speaker = str(parsed[3]) if len(parsed) > 3 else ""
                    party = str(parsed[4]) if len(parsed) > 4 else ""
                    chamber = str(parsed[5]) if len(parsed) > 5 else ""
                    dist_str = str(parsed[6]) if len(parsed) > 6 else "null"
                else:
                    continue
                
                # Limpiamos las comillas dobles
                if texto.startswith('"') and texto.endswith('"'):
                    texto = texto[1:-1]
                    
                score = float(dist_str) if dist_str != 'null' else 999.0
                party_clean = party if party != 'null' else ""
                
                chunks.append(RetrievedChunk(
                    chunk_id=c_id, 
                    intervention_id=i_id, 
                    text=texto, 
                    speaker="" if speaker == "null" else speaker,
                    party=party_clean,
                    chamber="" if chamber == "null" else chamber,
                    score=score
                ))
                
            except Exception as e:
                print(f"[ADVERTENCIA] Error parseando fila de MDB: {e}")
                
        return chunks

    def retrieve_as_context(self, question: str, k: Optional[int] = None) -> str:
        chunks = self.retrieve(question, k)
        if not chunks:
            return "No se encontró contexto relevante."
        return "\n\n---\n\n".join(c.to_context_str() for c in chunks)
