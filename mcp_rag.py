import requests

# URL do endpoint MCP do Crawl4AI (ajuste se necessário)
MCP_SERVER_ASK_URL = "http://localhost:11235/mcp/ask"

def get_mcp_context(query: str) -> str:
    """
    Chama o MCP Server via HTTP POST para obter documentos de RAG.
    """
    try:
        resp = requests.post(MCP_SERVER_ASK_URL, json={"query": query})
        resp.raise_for_status()
        data = resp.json()

        # Verifique o formato retornado
        # Pode ser uma lista de documentos ou uma estrutura com key "documents"
        docs = data.get("documents") or data.get("results") or data

        # Se for lista de strings ou dicts, converta:
        if isinstance(docs, list) and docs and isinstance(docs[0], dict):
            contents = [d.get("content", "") for d in docs]
        elif isinstance(docs, list) and isinstance(docs[0], str):
            contents = docs
        else:
            contents = [str(docs)]

        contexto = "\n\n".join(contents)
        return contexto.strip()

    except requests.RequestException as e:
        print(f"❌ Erro ao chamar MCP Server: {e}")
        return "[Erro ao recuperar contexto do Craw4AI]"