# Copyright 2025 Guilherme Rodovalho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

Atue como um analista de risco especializado em segurança de autoridades.

1. Identifique, de forma técnica e detalhada, as **principais ameaças e vulnerabilidades** presentes no cenário.
2. Analise ponto a ponto todos os elementos do contexto, buscando **extrair o máximo de riscos potenciais** e seus fatores agravantes.
3. Apresente também um **boletim climático para o dia do evento**, incluindo previsões relevantes que possam impactar as operações.

Utilize linguagem clara, objetiva e profissional. Seja minucioso em sua análise.
""")


PROMPT_ENGENHEIRO = PromptTemplate.from_template("""
CENÁRIO]
{cenario}

[ANÁLISE DO LOCAL]
{analise_local}

[ANÁLISE DE RISCO]
{analise_risco}

[MEMÓRIA DA CONVERSA]
{context}

Atue como um engenheiro de barreiras com expertise em segurança institucional.

1. Proponha **medidas mitigadoras físicas e tecnológicas** com base nas vulnerabilidades identificadas.
2. Justifique tecnicamente **a importância de cada barreira** proposta.
3. Sempre que possível, ofereça **alternativas tecnológicas** com prós e contras para subsidiar decisões estratégicas.

Pense ponto a ponto. Estruture sua resposta com clareza, incluindo justificativas técnicas e operacionais.
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

Atue como um **coordenador operacional** responsável por organizar e distribuir as equipes de segurança.

Pense ponto a ponto. Elabore um planejamento que contenha:

1. **Divisão dos turnos de serviço**, considerando as seguintes escalas possíveis:
   - 12h por 36h
   - 12h por 60h
   - 24h por 96h
   - 8h por dia com até 2h extras

2. **Distribuição das equipes**, especificando:
   - Quantitativo real de profissionais disponíveis;
   - Função de cada membro;
   - Alocação por área, função ou turno.

3. **Estrutura ideal da equipe**, conforme padrões recomendados:
   - Quantitativo ideal por função;
   - Justificativas para a composição ideal;
   - Comparações entre o cenário atual e o ideal, com análise de lacunas, riscos e pontos de atenção.

4. **Quadros comparativos ou tabelas** para facilitar a visualização entre realidade e ideal.

5. **Padronização de vestimentas e equipamentos**:
   - **Equipe Ostensiva**: Uniforme tático, armamento de dotação (pistola/fuzil/carabina), colete balístico e demais equipamentos conforme orientação da unidade.
   - **Equipe Velada**: Traje civil adequado ao ambiente, porte velado com pistola e equipamentos compatíveis com a missão.

Seja técnico, organizado e detalhado. O documento final deve refletir um planejamento completo e embasado.
""")

PROMPT_REVISOR = PromptTemplate.from_template("""
Pense ponto a ponto.
Atue como um **especialista em segurança de autoridades**, com experiência em planejamento de operações de alto risco.

Utilize os dados abaixo para redigir um **planejamento estruturado, técnico e extenso**, com uso de **títulos e subtítulos**.

[CENÁRIO]
{cenario}

[ANÁLISE DO LOCAL]
{analise_local}

[ANÁLISE DE RISCO]
{analise_risco}

[MEDIDAS]
{medidas}

[OPERAÇÃO]
{operacao}

[DOCUMENTAÇÃO DE APOIO]
{documentos_de_apoio}

Dê ênfase à análise do local e siga as diretrizes operacionais propostas pelo coordenador.

Utilize quadros comparativos ou tabelas sempre que for útil para a clareza.

Em relação à documentação de apoio:
- Inspire-se em planejamentos anteriores;
- Identifique os tópicos e subtópicos mais relevantes;
- Mantenha linguagem técnica e consistente;

Não economize palavras — o nível de detalhamento é essencial.

A entrega deve ser clara, completa e com forte embasamento técnico-operacional.
""")
PROMPT_FEEDBACK = PromptTemplate.from_template("""
Pense ponto a ponto                                               
Você é um revisor técnico especializado em segurança de autoridades.
Abaixo está o planejamento de segurança gerado:                                           

[DOCUMENTO]
{messages}
                                                                        
[REQUISIÇÃO DE REVISÃO]
{cenario} 
Com base nessa requisição, aponte ponto a ponto:
- O que precisa ser ajustado no documento;
- Por quê (justificativas técnicas);
- Quais seções devem ser revisadas ou reescritas;
- Se possível, sugestão do texto revisado ou orientação clara.

Se a requisição estiver incorreta ou irrelevante, explique o motivo.
Não economize palavras.
""")

# === Nodes ===
def analise_imagens_node(state):
    url = str(state["url"])
    print(f"\n\n\n\n {url}\n\n\n ")
    cenario = state["cenario"]
    if url == "N":
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
        ],
    }
    else:
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
    palavras_chave_revisao = [
        "revise", "revisar", "corrija", "corrigir", "ajuste", "ajustar",
        "modifique", "modificar", "melhore", "reescreva", "melhorar"
    ]
    if any(p in cenario for p in palavras_chave_revisao):
        print("revisao")
        clima = "sem informaçoes sobre o clima"
    else:
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
    avaliacao = resposta.invoke({"messages": state["messages"], "cenario": state["cenario"]})
    
    return {**state, "documento_final": avaliacao}

def condicional_revisao_final(state) -> str:
    query = state.get("cenario", "").lower()
    palavras_chave_revisao = [
        "revise", "revisar", "corrija", "corrigir", "ajuste", "ajustar",
        "modifique", "modificar", "melhore", "reescreva", "melhorar"
    ]
    if any(p in query for p in palavras_chave_revisao):
        return "engenheiro_feedback"
    return "__end__"


def documento_revisado_node(state):
    contexto = get_context(state["cenario"])
    resposta = PROMPT_REVISOR | image_llm | StrOutputParser()
    doc_final = resposta.invoke({
        "cenario": state["cenario"],
        "analise_local": state["analise_local"],
        "analise_risco": state["analise_risco"],
        "medidas": state["medidas"],
        "operacao": state["operacao"],
        "documentos_de_apoio": contexto
    })
    return {**state, "documento_final": doc_final}



# === Construindo o grafo ===
workflow = StateGraph(PlanejamentoState)
workflow.add_node("analise_local", analise_imagens_node)
workflow.add_node("analise_risco", analista_node)
workflow.add_node("engenheiro_feedback", engenheiro_feedback_node)
workflow.add_node("medidas", engenheiro_node)
workflow.add_node("operacao", coordenador_node)
workflow.add_node("documento", redator_node)
workflow.add_node("documento_revisado", documento_revisado_node)
workflow.set_entry_point("analise_local")
workflow.add_edge("analise_local","analise_risco")
workflow.add_edge("analise_risco", "medidas")
workflow.add_edge("medidas", "operacao")
workflow.add_edge("operacao", "documento")
workflow.add_conditional_edges(
    "documento",
    condicional_revisao_final,
    {
        "engenheiro_feedback": "engenheiro_feedback",
        "__end__": END
    }
)

# Fluxo após revisão:
workflow.add_edge("engenheiro_feedback", "documento_revisado")
workflow.add_edge("documento_revisado", END)

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