import streamlit as st
from save_vector_store import add_to_faiss
from core import responder
import base64
import html
import os
import time
import re
from session_id import get_next_session_id


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
st.set_page_config(page_title="JULIETT - Assistente Virtual da Polícia Judicial",  layout= "wide")


image_path = 'assets/images/logo.jpeg'
image_base64 = get_base64_image(image_path)
html_content = f"""
<div style="display: flex; align-items: center; justify-content: center;">
    <img src="data:image/png;base64,{image_base64}" style="width: 200px; height: auto; margin-right: 5px;">
    <div>
         <h1>JULIETT - Assistente Virtual da Polícia Judicial</h1>
         <p style="font-size: 28px; margin: 0; text-align: center;">Divisão de Polícia Judicial - Justiça Federal do Mato Grosso do Sul</p>
          <p style="font-size: 26px; margin: 0; text-align: center;">Inteligência Artificial Generativa</p>
    </div>
</div> 
"""

if len(st.session_state) == 0:
        st.session_state["messages"] = []

st.markdown(html_content, unsafe_allow_html=True)





def split(response):
    if isinstance(response, tuple):
        texto = response[0]  # Pega a resposta principal
    else:
        texto = response

    # Remove bloco <think>...</think>
    texto_limpo = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()

    # Converte \n\n e \n para quebras de linha reais
    texto_formatado = texto_limpo.replace("\\n", "\n").replace("\n\n", "\n\n")

    
#Entrega as respostas caractere por caractere   
    if "<think>" in texto_formatado:
        think_part = re.sub(r"<think>.*?</think>", "", str(texto_formatado), flags=re.DOTALL).strip()
        
        caracteres = [char for char in think_part]
        for char in caracteres:     
            yield char
            time.sleep(0.01)
    else:
        caracteres = [char for char in str(texto_formatado)]
        for char in caracteres:     
            yield char
            time.sleep(0.01)



# Interface do Streamlit

# Campo de entrada para a pergunta do usuário
#resposta_placeholder = st.empty()

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

#resposta_placeholder.container(height=300, border= False)
input = st.text_area(label="Faça uma pergunta:")

# Botão para enviar a pergunta
if "generated_response" not in st.session_state:
    st.session_state.generated_response = None
if st.button("Enviar"):
    
    if input:
        query = html.escape(input)
        response = responder(query, st.session_state["session_id"])
        
        st.session_state.user_input =""
        st.session_state["messages"].append((query, True))
        st.session_state["messages"].append((response, False))
     # Obter resposta       
        
        #st.write("Resposta:")
        resposta = response
        st.session_state.generated_response = response
        #mp3_fp = asyncio.run(texto_para_audio(resposta))
        
       # with resposta_placeholder.container(height=300, border= True):
        if st.session_state.generated_response:
            st.write("Resposta: ")
            st.write_stream(split(st.session_state.generated_response))
        
       

    else:
            st.warning("Por favor, insira uma pergunta.")
            
st.markdown("<p style='padding-top:10px'> </p>", unsafe_allow_html=True)

with st.expander("CLIQUE AQUI PARA VER O HISTÓRICO DA CONVERSA"):
   # display_messages()
   print("olá")

st.markdown("<p style='padding-top:10px'> </p>", unsafe_allow_html=True)


with st.expander("CLIQUE AQUI PARA TRABALHAR COM ARQUIVOS LOCAIS"):
    file = []
    if 'temp_file_path' not in st.session_state:
        st.session_state.temp_file_path = None

    upload_file = st.file_uploader("Escolha um arquivo para enviar para a Juliett", type=["pdf"])

    if upload_file is not None:
        save_directory = f".venv/data/pdf_temp"
        name_file = 'temp.pdf'
        caminho_arquivo = os.path.join(save_directory, name_file)
        if os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)
            os.makedirs(save_directory, exist_ok=True)
            file_path = os.path.join(save_directory, name_file)
            st.session_state.temp_file_path = file_path
            with open(file_path, "wb") as f:
                f.write(upload_file.getbuffer())
        else:
            os.makedirs(save_directory, exist_ok=True)
            file_path = os.path.join(save_directory, name_file)
            st.session_state.temp_file_path = file_path
            with open(file_path, "wb") as f:
                f.write(upload_file.getbuffer())
        #falta criar a função para deletar o arquivo criado

   

    upload_rag = st.file_uploader("Escolha um arquivo para adicionar a base de dados da Juliett", type=["pdf"])
    
    if st.button('Adicionar'):
        if upload_rag is not None:
            
            if upload_rag.name not in file:

                save_dir = "/Users/guilherme-rodovalho/Documents/RAG"
                save_path = os.path.join(save_dir, upload_rag.name)
            #Salve o arquivo
                with open(save_path, 'wb') as f:
                    f.write(upload_rag.getbuffer())
                    
                    retorno = add_to_faiss(save_dir, save_path,)
                    if "sucesso" in retorno:
                        st.success(retorno)
                    
                    else:
                        st.warning(retorno)
                    file.append(upload_rag.name)
            
        else:
            st.warning("Insira um arquivo primeiro")

    


st.markdown("<p style='padding-top:10px'> </p>", unsafe_allow_html=True)    
if 'session_id' not in st.session_state:
   # next_session_id = get_next_session_id
   table = 'sessions'
   st.session_state['session_id'] = str(get_next_session_id(table))



