"""Lê o PDF do manual, divide em trechos (1 por slide), gera embeddings e grava no ChromaDB.

Rode UMA vez (e de novo sempre que o PDF mudar):
    python ingest.py data/manual_estagio.pdf
"""
import re
import sys

import chromadb
import pymupdf  # pip install pymupdf

import rag

MIN_CHARS = 120  # slides com menos texto que isso (capa, telas só com imagem) são ignorados


def _limpa(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.replace("\u00a0", " ")).strip()


def _eh_titulo(txt: str) -> bool:
    letras = [c for c in txt if c.isalpha()]
    return len(letras) >= 6 and all(c.isupper() for c in letras)


def carregar_chunks(caminho_pdf: str) -> list[dict]:
    """Um chunk por slide: cada slide do manual já é uma unidade de assunto completa."""
    chunks = []
    for i, page in enumerate(pymupdf.open(caminho_pdf), start=1):
        # blocos de texto na ordem de leitura (de cima para baixo): o título vem primeiro
        blocos = [_limpa(b[4]) for b in page.get_text("blocks", sort=True) if b[6] == 0]
        blocos = [b for b in blocos if b and b != str(i) and b.lower() != "público"]  # nº do slide / rodapé

        titulo = blocos[0] if blocos and _eh_titulo(blocos[0]) else f"Slide {i}"
        corpo_blocos = blocos[1:] if titulo != f"Slide {i}" else blocos
        corpo = "\n".join(corpo_blocos)

        if len(corpo) < MIN_CHARS:
            print(f"  slide {i:>2}: ignorado (pouco texto)")
            continue
        chunks.append({"id": f"slide-{i}", "pagina": i, "titulo": titulo, "texto": f"{titulo}\n{corpo}"})
    return chunks


def main(caminho_pdf: str):
    chunks = carregar_chunks(caminho_pdf)
    print(f"{len(chunks)} trechos prontos. Gerando embeddings com '{rag.EMBED_MODEL}'...")

    embeddings = []
    for i in range(0, len(chunks), 16):  # em lotes, para não pesar
        embeddings += rag.embed([c["texto"] for c in chunks[i:i + 16]], tipo="doc")

    client = chromadb.PersistentClient(path=rag.DB_DIR)
    try:  # recria do zero para não duplicar nada ao reexecutar
        client.delete_collection(rag.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(
        name=rag.COLLECTION,
        metadata={"hnsw:space": "cosine"},  # distância cosseno (0 = idêntico)
        embedding_function=None,            # os embeddings são gerados por nós, via Ollama
    )
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["texto"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"pagina": c["pagina"], "titulo": c["titulo"]} for c in chunks],
    )
    print(f"Pronto! {col.count()} trechos salvos em '{rag.DB_DIR}/'.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/manual_estagio.pdf")
