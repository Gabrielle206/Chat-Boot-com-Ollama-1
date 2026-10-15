"""Funções compartilhadas do RAG: embeddings (Ollama) + busca no ChromaDB.

Uso rápido para testar a busca no terminal:
    python rag.py "como cadastro uma empresa nova?"
"""
import os
import sys

import chromadb
import ollama

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")   # ollama pull bge-m3
DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "manual_estagio"

TOP_K = 4            # quantos trechos entram no prompt
MAX_DISTANCE = 0.55  # distância cosseno máxima aceita (calibre com o CLI abaixo)


def _prefixo(tipo: str) -> str:
    # O nomic-embed-text exige prefixos; o bge-m3 não precisa de nenhum.
    if "nomic" in EMBED_MODEL:
        return "search_document: " if tipo == "doc" else "search_query: "
    return ""


def embed(textos: list[str], tipo: str = "doc") -> list[list[float]]:
    """Gera os embeddings via Ollama. tipo = 'doc' (trechos) ou 'query' (pergunta)."""
    resp = ollama.embed(model=EMBED_MODEL, input=[_prefixo(tipo) + t for t in textos])
    return resp["embeddings"]


def get_client():
    return chromadb.PersistentClient(path=DB_DIR)


def get_collection():
    return get_client().get_collection(COLLECTION)


def retrieve(pergunta: str, k: int = TOP_K, max_distance: float = MAX_DISTANCE):
    """Retorna os trechos mais parecidos com a pergunta (já filtrados por distância).

    Cada item: {"texto", "pagina", "titulo", "distancia"}, do mais para o menos relevante.
    """
    col = get_collection()
    res = col.query(
        query_embeddings=embed([pergunta], tipo="query"),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= max_distance:
            hits.append(
                {"texto": doc, "pagina": meta["pagina"], "titulo": meta["titulo"], "distancia": dist}
            )
    return hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "como cadastro uma empresa nova?"
    # max_distance=9 para ver TODOS os resultados e poder calibrar o limite
    for h in retrieve(q, k=6, max_distance=9):
        print(f"dist={h['distancia']:.3f}  slide {h['pagina']:>2}  {h['titulo']}")
