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
from config import OPENAI_KEY
from get_rag_context import get_relevant_documents
from langchain_openai import ChatOpenAI
from get_weather import get_weather
import re, json


# === Configurações iniciais ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
memory = MemorySaver()

image_llm = ChatOpenAI(
    model = "gpt-4o",
    api_key = OPENAI_KEY
)
llm = ChatGroq(api_key=API_KEY, model="qwen/qwen3-32b")
llm1 = ChatGroq(api_key=API_KEY, model="llama-3.3-70b-versatile", temperature= 0.1)
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
    index = "rag_index"
    documents = get_relevant_documents(input=query, path=index)
    return documents

# === Estado ===
class PlanejamentoState(TypedDict):
    cenario: str
    analise_local: str
    analise_risco: str
    medidas: str
    operacao: str
    documento_final: str
    messages: List[Dict[str, str]]
    context: str
    url: str
    clima: str
    

# === Prompts ===

PROMPT_ANALISTA = PromptTemplate.from_template("""
[CENÁRIO]
{cenario}
[MEMÓRIA DA CONVERSA]
{context}

Como analista de risco, identifique as principais ameaças e vulnerabilidades.
Apresente um release do clima para o dia do evento
Pense ponto a ponto em todos os detalhos existentes no cenário para identificar o máximo de ameaças e vulnerabilidades.                                               
""")

PROMPT_ENGENHEIRO = PromptTemplate.from_template("""
[CENÁRIO]
{cenario}
[ANÁLISE DO LOCAL]
{analise_local}                                                                                                  
[ANÁLISE DE RISCO]
{analise_risco}
[MEMÓRIA DA CONVERSA]
{context}
Pense ponto a ponto
Como engenheiro de barreiras, proponha medidas mitigadoras físicas e tecnológicas.
""")

PROMPT_COORDENADOR = PromptTemplate.from_template("""
[CENÁRIO]
{cenario}
[ANÁLISE DO LOCAL]
{analise_local}  
[ANÁLISE DE RISCO]
{analise_risco}
[MEDIDAS]
{medidas}
[MEMÓRIA DA CONVERSA]
{context}
Pense ponto a ponto
Como coordenador operacional, planeje turnos, pontos de controle, divisão das equipes, qual vestimenta e equipamentos cada equipe deve utilizar e rotinas.
Equipe de segurança ostensiva: Uniforme e equipamentos: Uniforme tático, armamento de dotação pistola/fuzil/carabina, colete balístico e equipamento conforme orientação de sua unidade.
Equipe de segurança Velada: Traje civil comum ao ambiente, armamento de dotação pistola com o porte velado e equipamento conforme orientação de sua unidade. 
""")

PROMPT_REVISOR = PromptTemplate.from_template("""
Pense ponto a ponto
Atue como um especialista em segurança de autoridades.
Use os dados a seguir para gerar um planejamento estruturado:

Cenário: {cenario}
Análise do local: {analise_local}                                              
Análise de risco: {analise_risco}
Medidas: {medidas}
Operação: {operacao}
documentação de apoio: {documentos_de_apoio}                                              
                                              
Dê relevancia a análise do local
Crie um documento extenso, claro e técnico com títulos e subtítulos, seja bem detalhista.
                                              
Em relação a documentação de apoio, se inspire nos planejamentos existentes, percebendo os tópicos e subtopicos relevantes.
Não economize palavaras
""")
PROMPT_FEEDBACK = PromptTemplate.from_template("""
Pense ponto a ponto                                               
Você é um engenheiro de segurança. A seguir está a análise de risco feita pelo analista:

"{analise_risco}"

Ela é suficiente para que você proponha medidas técnicas e operacionais?

Responda apenas com "sim" ou "não" e uma frase explicativa.
""")

# === Nodes ===
def analise_imagens_node(state):
    url = str(state["url"])
    print(f"\n\n\n\n {url}\n\n\n ")
    cenario = state["cenario"]
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Você é um especialista em segurança de dignitários altamente treinado. "
                    "1. Faça uma análise da imagem recebida e identifique riscos, perímetros, acessos, rotas de fuga, pontos de vigilância e outros elementos relevantes.\n"
                    "2. Em seguida, com base na frase fornecida pelo usuário, extraia o nome da cidade e a data do evento no formato JSON com os campos 'cidade' e 'data'.\n"
                    f"Frase do usuário: {cenario}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{url}",
                },
            },
        ],
    }

    resposta = image_llm.invoke([message])  # dev
    texto = resposta.content
    print(f"\n\n ANÁLISE IMAGEM \n\n {texto} \n\n")
    try:
        match = re.search(r'\{.*"cidade".*?\}', texto, re.DOTALL)
        local_info = json.loads(match.group()) if match else {}
    except Exception as e:
        print("❌ Erro ao extrair JSON de cidade/data:", e)
        local_info = {}
    clima = get_weather(local_info.get("cidade"))
    
    print(f"\n\n release do clima \n\n {clima} \n\n")
    resposta = f"{texto} segue o release do clima para ser analisado {clima}"
    return{**state, "analise_local": resposta, "clima": clima}
    
def analista_node(state):
    contexto_memoria = filtrar_memoria_relevante(state["cenario"], state.get("messages", []))
    #clima = get_weather(state["lat"], state["lon"], state["data"])
    
    resposta = PROMPT_ANALISTA | llm | StrOutputParser()
    cenario = f"{state['cenario']} {state['analise_local']}"
    analise = resposta.invoke({"cenario": cenario, "context": contexto_memoria})
    return {**state, "analise_risco": analise, "context": contexto_memoria}

def engenheiro_node(state):
    resposta = PROMPT_ENGENHEIRO | llm1 | StrOutputParser()
    
    medidas = resposta.invoke({"cenario": state["cenario"], "analise_local": state["analise_local"],"analise_risco": state["analise_risco"], "context": state["context"]})
    return {**state, "medidas": medidas}

def coordenador_node(state):
    resposta = PROMPT_COORDENADOR | llm2 | StrOutputParser()
    
    operacao = resposta.invoke({"cenario": state["cenario"], "analise_local": state["analise_local"],"analise_risco": state["analise_risco"], "medidas": state["medidas"], "context": state["context"]})
    return {**state, "operacao": operacao}

def redator_node(state):
    contexto = get_context(state["cenario"])
    
    resposta = PROMPT_REVISOR | image_llm | StrOutputParser()
    doc_final = resposta.invoke({"cenario": state["cenario"],"analise_local":state["analise_local"], "analise_risco": state["analise_risco"], "medidas": state["medidas"], "operacao": state["operacao"],"documentos_de_apoio": contexto})
    print(f"\n\n\n DOC FINAL = {state['analise_risco']}")
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
workflow.add_node("analise_local", analise_imagens_node)
workflow.add_node("analise_risco", analista_node)
workflow.add_node("engenheiro_feedback", engenheiro_feedback_node)
workflow.add_node("medidas", engenheiro_node)
workflow.add_node("operacao", coordenador_node)
workflow.add_node("documento", redator_node)
workflow.set_entry_point("analise_local")
#workflow.set_entry_point("analise_risco")
workflow.add_edge("analise_local","analise_risco")
workflow.add_edge("analise_risco", "medidas")

#workflow.add_conditional_edges("engenheiro_feedback", lambda state: "analise_risco" if state["refazer_analise"] else "medidas")
workflow.add_edge("medidas", "operacao")
workflow.add_edge("operacao", "documento")
workflow.add_edge("documento", END)

graph = workflow.compile(checkpointer=memory)

# === Função para executar com histórico ===
def responder(cenario: str, url: str,thread_id: str = "default", mensagens: List[Dict[str, str]] = []) -> str:
    print(f"\n\n\n {url} \n\n")
    state = {
        "cenario": cenario,
        "analise_local" : "",
        "analise_risco": "",
        "medidas": "",
        "operacao": "",
        "documento_final": "",
        "messages": mensagens,
        "context": "",
        "url": url,
        "clima": ""
        
    }
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state, config=config)
    imprimir = f"{result['documento_final']} \n Release do CLima: \n {state['clima']}"
    print(imprimir)
    return result["documento_final"]