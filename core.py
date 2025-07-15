# Copyright 2025 Guilherme Felipe Breetz Rodovalho 
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



from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.chat_models.base import BaseChatModel
from config import API_KEY
from get_rag_context import get_relevant_documents
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver


TOKENIZERS_PARALLELISM= True

memory = MemorySaver()
llm: BaseChatModel = ChatGroq(api_key= API_KEY, model= "qwen/qwen3-32b")
embedder = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

def get_context(query: str) -> str:

    index = "faiss_index"
    documents = get_relevant_documents(input=query, path=index)

    print(f"\n\n os documentos são \n\n {documents}")
    return documents

# Extrai mensagens anteriores como texto
def extrair_chunks_mensagens(messages: List[Dict[str, str]]) -> List[str]:
    chunks = []
    for m in messages:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        chunks.append(f"{role}: {m['content']}")
    return chunks

# Indexa a memória da conversa
def indexar_memoria(messages: List[Dict[str, str]]) -> FAISS:
    chunks = extrair_chunks_mensagens(messages)
    docs = [Document(page_content=c) for c in chunks]
    return FAISS.from_documents(docs, embedder)

# Recupera apenas os trechos relevantes da memória para a pergunta
def filtrar_memoria_relevante(pergunta: str, messages: List[Dict[str, str]], k: int = 2) -> str:
    if not messages:
        return ""
    vectorstore = indexar_memoria(messages)
    docs_relevantes = vectorstore.similarity_search(pergunta, k=k)
    return "\n".join([doc.page_content for doc in docs_relevantes])

class GraphState(TypedDict):
    messages: List[Dict[str, str]]
    input: str
    response: str
    context: str

def construir_grafo():
    def recuperar_node(state: GraphState) -> GraphState:
        pergunta = state["input"]

        memoria_conversacional = filtrar_memoria_relevante(pergunta, state.get("messages", []), k=2)

        contexto_docs = get_context(pergunta)
        contexto_geral = f"[MEMÓRIA DE CONVERSA]\n{memoria_conversacional}\n\n[BASE DE CONHECIMENTO]\n{contexto_docs}"

        print(f"\n\n[🔍 Contexto geral]\n{contexto_geral}")
        return {**state, "context": contexto_geral}

    def gerar_node(state: GraphState) -> GraphState:
        prompt = PromptTemplate.from_template(
            "Use as informações a seguir para responder com clareza. Se nas informações não existir nada que auxilie na resposta, peça mais informações ao usuário sempre mantendo uma conversa fluida e sem mencionar a palavra contexto\n\n{context}\n\nPergunta:\n{question}\n\nResposta:"
        )
        parser = StrOutputParser()
        chain = prompt | llm | parser

        resposta = chain.invoke({
            "context": state["context"],
            "question": state["input"]
        })

        # Atualiza mensagens
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": state["input"]})
        messages.append({"role": "assistant", "content": resposta})

        return {
            "input": state["input"],
            "response": resposta,
            "messages": messages,
            "context": state["context"]
        }

    builder = StateGraph(GraphState)
    builder.add_node("recuperar_contexto", recuperar_node)
    builder.add_node("gerar_resposta", gerar_node)
    builder.set_entry_point("recuperar_contexto")
    builder.add_edge("recuperar_contexto", "gerar_resposta")
    builder.add_edge("gerar_resposta", END)

    return builder.compile(checkpointer= memory)

graph = construir_grafo()

# Mantém histórico automaticamente via thread_id
def responder(pergunta: str, thread_id: str = "default") -> str:
    print(f"\n\n o thread_id é = {thread_id}")
    state = {
        "input": pergunta,
       # "messages": [],      # Ignorado se thread_id já tiver estado
        "response": "",
        "context": ""
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state, config=config)

    return result["response"], result["messages"]


