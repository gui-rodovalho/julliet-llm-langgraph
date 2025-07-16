# agentes_planejamento_seg.py adaptado com core LangGraph + RAG + memória

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
import os
from config import API_KEY
from get_rag_context import get_relevant_documents

# === Configurações iniciais ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
memory = MemorySaver()

llm = ChatGroq(api_key=API_KEY, model="qwen/qwen3-32b")
llm1 = ChatGroq(api_key=API_KEY, model="llama-3.3-70b-versatile")
llm2 = ChatGroq(api_key=API_KEY, model="deepseek-r1-distill-llama-70b")
embedder = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

# === Utilitários ===
def extrair_chunks_mensagens(messages: List[Dict[str, str]]) -> List[str]:
    chunks = []
    for m in messages:
        role = "Usuário" if m["role"] == "user" else "Agente"
        chunks.append(f"{role}: {m['content']}")
    return chunks

def indexar_memoria(messages: List[Dict[str, str]]) -> FAISS:
    chunks = extrair_chunks_mensagens(messages)
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embedder)

def filtrar_memoria_relevante(pergunta: str, messages: List[Dict[str, str]], k: int = 2) -> str:
    if not messages:
        return ""
    vectorstore = indexar_memoria(messages)
    docs_relevantes = vectorstore.similarity_search(pergunta, k=k)
    return "\n".join([doc.page_content for doc in docs_relevantes])

def get_context(query: str) -> str:
    index = "faiss_index"
    documents = get_relevant_documents(input=query, path=index)
    return documents

# === Estado ===
class PlanejamentoState(TypedDict):
    cenario: str
    analise_risco: str
    medidas: str
    operacao: str
    documento_final: str
    messages: List[Dict[str, str]]
    context: str

# === Prompts ===
PROMPT_ANALISTA = PromptTemplate.from_template("""
[CENÁRIO]
{cenario}
[MEMÓRIA DA CONVERSA]
{context}

Como analista de risco, identifique as principais ameaças e vulnerabilidades.
""")

PROMPT_ENGENHEIRO = PromptTemplate.from_template("""
[CENÁRIO]
{cenario}
[ANÁLISE DE RISCO]
{analise_risco}
[MEMÓRIA DA CONVERSA]
{context}

Como engenheiro de barreiras, proponha medidas mitigadoras físicas e tecnológicas.
""")

PROMPT_COORDENADOR = PromptTemplate.from_template("""
[CENÁRIO]
{cenario}
[ANÁLISE DE RISCO]
{analise_risco}
[MEDIDAS]
{medidas}
[MEMÓRIA DA CONVERSA]
{context}

Como coordenador operacional, planeje turnos, pontos de controle, divisão das equipes, qual vestimenta e equipamentos cada equipe deve utilizar e rotinas.
""")

PROMPT_REVISOR = PromptTemplate.from_template("""
Atue como um especialista em segurança de autoridades.
Use os dados a seguir para gerar um planejamento estruturado:

Cenário: {cenario}
Análise de risco: {analise_risco}
Medidas: {medidas}
Operação: {operacao}
documentação de apoio: {documentos_de_apoio}                                              
                                              

Crie um documento claro e técnico com títulos e subtítulos, seja bem detalhista.
""")
PROMPT_FEEDBACK = PromptTemplate.from_template("""
Você é um engenheiro de segurança. A seguir está a análise de risco feita pelo analista:

"{analise_risco}"

Ela é suficiente para que você proponha medidas técnicas e operacionais?

Responda apenas com "sim" ou "não" e uma frase explicativa.
""")

# === Nodes ===
def analista_node(state):
    contexto_memoria = filtrar_memoria_relevante(state["cenario"], state.get("messages", []))
    resposta = PROMPT_ANALISTA | llm | StrOutputParser()
    analise = resposta.invoke({"cenario": state["cenario"], "context": contexto_memoria})
    return {**state, "analise_risco": analise, "context": contexto_memoria}

def engenheiro_node(state):
    resposta = PROMPT_ENGENHEIRO | llm | StrOutputParser()
    medidas = resposta.invoke({"cenario": state["cenario"], "analise_risco": state["analise_risco"], "context": state["context"]})
    return {**state, "medidas": medidas}

def coordenador_node(state):
    resposta = PROMPT_COORDENADOR | llm2 | StrOutputParser()
    operacao = resposta.invoke({"cenario": state["cenario"], "analise_risco": state["analise_risco"], "medidas": state["medidas"], "context": state["context"]})
    return {**state, "operacao": operacao}

def redator_node(state):
    contexto = get_context(state["cenario"])
    
    resposta = PROMPT_REVISOR | llm1 | StrOutputParser()
    doc_final = resposta.invoke({"cenario": state["cenario"], "analise_risco": state["analise_risco"], "medidas": state["medidas"], "operacao": state["operacao"], "documentos_de_apoio": contexto})
    print(f"\n\n\n DOC FINAL = {state["analise_risco"]}")
    return {**state, "documento_final": doc_final}

def engenheiro_feedback_node(state):
    resposta = PROMPT_FEEDBACK | llm | StrOutputParser()
    avaliacao = resposta.invoke({"analise_risco": state["analise_risco"]})
    if "não" in avaliacao.lower():
        return {**state, "refazer_analise": True}
    else:
        return {**state, "refazer_analise": False}

# === Construindo o grafo ===
workflow = StateGraph(PlanejamentoState)
workflow.add_node("analise_risco", analista_node)
workflow.add_node("engenheiro_feedback", engenheiro_feedback_node)
workflow.add_node("medidas", engenheiro_node)
workflow.add_node("operacao", coordenador_node)
workflow.add_node("documento", redator_node)
workflow.set_entry_point("analise_risco")
workflow.add_edge("analise_risco", "engenheiro_feedback")
workflow.add_conditional_edges("engenheiro_feedback", lambda state: "analise_risco" if state["refazer_analise"] else "medidas")
workflow.add_edge("medidas", "operacao")
workflow.add_edge("operacao", "documento")
workflow.add_edge("documento", END)

graph = workflow.compile(checkpointer=memory)

# === Função para executar com histórico ===
def responder(cenario: str, thread_id: str = "default", mensagens: List[Dict[str, str]] = []) -> str:
    state = {
        "cenario": cenario,
        "analise_risco": "",
        "medidas": "",
        "operacao": "",
        "documento_final": "",
        "messages": mensagens,
        "context": ""
    }
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state, config=config)
    return result["documento_final"]