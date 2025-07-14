from typing import TypedDict, List, Dict, Literal
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.chat_models.base import BaseChatModel
from config import API_KEY

llm: BaseChatModel = ChatGroq(api_key= API_KEY, model= "qwen/qwen3-32b")

class GraphState(TypedDict):
    messages: List[Dict[str, str]]
    input: str
    response: str

# nó LLM

def chat_with_llm(state: GraphState) -> GraphState:
    messages = state["messages"]
    input_text = state["input"]

    #adiciona a mensagem do usuário ao histórico
    messages.append({"role": "user", "content": input_text})

    # Converte oara ibjetos HumanMessage/AIMessage (se necesário)

    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content= msg["content"]))

    #Chamada à LLM

    resposta = llm.invoke(lc_messages)
    resposta_str = resposta.content if hasattr(resposta, "content") else resposta

    # Adiciona a resposta da IA ao histórico
    messages.append({"role": "assistant", "content": resposta_str})

    # Retorna estado atualizado
    return {
        "messages": messages,
        "input": input_text,
        "response": resposta_str
    }

# --- construir o grafo

builder = StateGraph(GraphState)
builder.add_node("chat", chat_with_llm)
builder.set_entry_point("chat")
builder.add_edge("chat", END)

graph = builder.compile()

# ----------- LOOP DE INTERAÇÃO -----------

if __name__ == "__main__":
    print("🧠 Chat com LangGraph (com histórico controlado) — Digite 'sair' para encerrar\n")
    
    state: GraphState = {"messages": [], "input": "", "response": ""}
    
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break

        state["input"] = user_input
        state = graph.invoke(state)
        print("Assistente:", state["response"])