import re

import streamlit as st
import ollama

import rag

st.set_page_config(page_title="Chat Local com Ollama", page_icon="🤖")
st.title("Chatbot Local - Tutor de estágio")

model_name = "tutor-qwen"
st.caption(f"Modelo: {model_name} | Base: Manual de Estágio Obrigatório 2026/2 (RAG)")

SEM_RESPOSTA = (
    "Não encontrei essa informação no manual de estágio obrigatório. "
    "Posso ajudar com dúvidas sobre cadastro de empresa, supervisores, termo de compromisso, "
    "prazos e seguro do estágio. Se a dúvida for outra, procure o seu polo ou a coordenação do curso."
)

PROMPT_CLASSIFICA = """Classifique a mensagem de um aluno em uma palavra.
CONVERSA = saudação, despedida, agradecimento, "tudo bem?", elogio ou papo social, SEM pedir nenhuma informação.
PERGUNTA = qualquer pedido de informação, dúvida, explicação ou ajuda, sobre qualquer assunto.

Exemplos:
"oi" -> CONVERSA
"e aí, tudo bem?" -> CONVERSA
"valeu, ajudou muito" -> CONVERSA
"como cadastro uma empresa?" -> PERGUNTA
"qual a capital da França?" -> PERGUNTA
"e para enfermagem?" -> PERGUNTA

Mensagem: "{msg}"
Responda apenas CONVERSA ou PERGUNTA."""

PROMPT_CONVERSA = """A mensagem do aluno é só conversa social (saudação, agradecimento, "tudo bem?"), não é uma pergunta de informação.
Responda como uma pessoa simpática responderia: de forma natural e curta (1 frase, no máximo 2), em português, retribuindo o cumprimento.
Se fizer sentido, termine perguntando de forma genérica como pode ajudar (por exemplo: "Como posso te ajudar?").
NÃO mencione estágio, manual, documentos, prazos nem qualquer assunto específico: o aluno ainda não falou de nenhum.
NÃO dê nenhuma informação, dado ou procedimento nesta resposta.

MENSAGEM DO ALUNO: {pergunta}"""

PROMPT_RAG = """Responda à pergunta do aluno usando SOMENTE os trechos do manual abaixo.
Se os trechos não trazem a resposta, diga que não encontrou no manual.
Não escreva a fonte nem cite números de slide: o sistema faz isso.

TRECHOS DO MANUAL:
{contexto}

PERGUNTA DO ALUNO: {pergunta}"""

if st.button("Limpar conversa"):
    st.session_state["messages"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ---------- RAG ----------
def montar_consulta(pergunta):
    """Pergunta curta (ex.: 'e para enfermagem?') perde o assunto: junta com a pergunta anterior."""
    if len(pergunta) < 40:
        anteriores = [m["content"] for m in st.session_state["messages"] if m["role"] == "user"]
        if anteriores:
            return f"{anteriores[-1]} {pergunta}"
    return pergunta


def classificar(mensagem):
    """'conversa' ou 'pergunta'. Na dúvida, trata como pergunta (que só responde com o manual)."""
    if len(mensagem) > 60:  # mensagem longa: claramente uma pergunta
        return "pergunta"
    try:
        r = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Você é um classificador. Responda com uma única palavra."},
                {"role": "user", "content": PROMPT_CLASSIFICA.format(msg=mensagem)},
            ],
            options={"temperature": 0, "num_predict": 5},
        )
        return "conversa" if "CONVERSA" in r["message"]["content"].upper() else "pergunta"
    except Exception:
        return "pergunta"


def gerar(mensagens, placeholder, **opcoes):
    """Faz o streaming da resposta do modelo. Retorna (texto, métricas)."""
    texto, metrics = "", None
    stream = ollama.chat(model=model_name, messages=mensagens, stream=True, options=opcoes or None)
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            texto += chunk["message"]["content"]
            placeholder.markdown(texto + "▌")
        if chunk.get("done", False):
            p_dur = chunk.get("prompt_eval_duration", 0)
            e_dur = chunk.get("eval_duration", 0)
            p_count = chunk.get("prompt_eval_count", 0)
            e_count = chunk.get("eval_count", 0)
            metrics = {
                "model": model_name,
                "total_duration": chunk.get("total_duration", 0) / 1e9,
                "load_duration": chunk.get("load_duration", 0) / 1e6,
                "prompt_eval_count": p_count,
                "prompt_eval_duration": p_dur / 1e9,
                "prompt_eval_rate": (p_count / (p_dur / 1e9)) if p_dur > 0 else 0,
                "eval_count": e_count,
                "eval_duration": e_dur / 1e9,
                "eval_rate": (e_count / (e_dur / 1e9)) if e_dur > 0 else 0,
            }
    return texto, metrics


def historico_anterior():
    return [
        {"role": m["role"], "content": m["content"]} for m in st.session_state["messages"][:-1]
    ][-6:]


def remover_fonte_do_modelo(texto):
    """Se o modelo escreveu uma linha 'Fonte: ...' por conta própria, tira (a fonte real é montada abaixo)."""
    linhas = [l for l in texto.splitlines() if not re.match(r"^\W*fonte\s*:", l.strip(), re.I)]
    return "\n".join(linhas).rstrip()


def montar_contexto(hits):
    return "\n\n".join(f"[Slide {h['pagina']} - {h['titulo']}]\n{h['texto']}" for h in hits)


def texto_fontes(hits):
    slides = sorted({h["pagina"] for h in hits})
    return "**Fonte:** Manual de Estágio Obrigatório 2026/2 — slide(s) " + ", ".join(map(str, slides))


def mostrar_trechos(hits):
    with st.expander("Trechos recuperados (RAG)"):
        for h in hits:
            st.markdown(f"**Slide {h['pagina']}** — {h['titulo']}  \n`distância: {h['distancia']:.3f}`")
            st.text(h["texto"])


# ---------- Métricas ----------
def mostrar_metricas(m):
    with st.expander("Informações Técnicas"):
        st.code(
            f"Modelo:                       {m['model']}\n"
            f"Tempo total:                  {m['total_duration']:.2f} s\n"
            f"Tempo para carregar o modelo: {m['load_duration']:.1f} ms\n"
            f"\n"
            f"Tokens lidos (prompt):        {m['prompt_eval_count']}\n"
            f"Tempo de leitura:             {m['prompt_eval_duration']:.3f} s\n"
            f"Velocidade de leitura:        {m['prompt_eval_rate']:.2f} tokens/s\n"
            f"\n"
            f"Tokens gerados (resposta):    {m['eval_count']}\n"
            f"Tempo de geração:             {m['eval_duration']:.3f} s\n"
            f"Velocidade de geração:        {m['eval_rate']:.2f} tokens/s",
            language=None,
        )


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            mostrar_trechos(message["hits"])
        if message.get("metrics"):
            mostrar_metricas(message["metrics"])

if prompt := st.chat_input("Digite a sua mensagem..."):
    consulta = montar_consulta(prompt)  # antes de guardar a pergunta atual no histórico
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        metrics = None
        hits = []

        try:
            if classificar(prompt) == "conversa":
                # Papo social: o modelo responde livremente, mas sem informar nada
                mensagens = historico_anterior() + [
                    {"role": "user", "content": PROMPT_CONVERSA.format(pergunta=prompt)}
                ]
                full_response, metrics = gerar(mensagens, placeholder, num_predict=150)
                full_response = remover_fonte_do_modelo(full_response)
                placeholder.markdown(full_response)
                if metrics:
                    mostrar_metricas(metrics)
            else:
                # Pergunta de informação: só responde com o que o manual traz
                try:
                    hits = rag.retrieve(consulta)
                except Exception as e:
                    st.error(
                        f"Não consegui consultar a base vetorial: {e}\n\n"
                        "Rode `python ingest.py data/manual_estagio.pdf` e confira se o modelo "
                        f"de embeddings está instalado (`ollama pull {rag.EMBED_MODEL}`)."
                    )
                    st.stop()

                if not hits:
                    # Nada parecido no manual: nem chama o LLM (evita resposta inventada)
                    full_response = SEM_RESPOSTA
                    placeholder.markdown(full_response)
                else:
                    mensagens = historico_anterior() + [
                        {
                            "role": "user",
                            "content": PROMPT_RAG.format(contexto=montar_contexto(hits), pergunta=prompt),
                        }
                    ]
                    full_response, metrics = gerar(mensagens, placeholder)
                    # A fonte é montada pelo código (não pelo modelo), para nunca ser inventada
                    full_response = remover_fonte_do_modelo(full_response) + "\n\n" + texto_fontes(hits)
                    placeholder.markdown(full_response)
                    mostrar_trechos(hits)
                    if metrics:
                        mostrar_metricas(metrics)

            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response, "hits": hits, "metrics": metrics}
            )

        except Exception as e:
            placeholder.error(f"Erro ao ligar ao Ollama: {e}")