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

# view vinculada ao core_agents.py
# sistema de multi agentes de IA

import streamlit as st
from save_vector_store import add_to_faiss
from core_agents import responder
import base64
import io
from PIL import Image
import html
import os
import time
import re
from session_id import get_next_session_id

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
#configurações do layout da aba do navegador
    
st.set_page_config(page_title="JULIETT - Assistente Virtual da Polícia Judicial",  layout= "wide")

# configurações de layout do Cabeçalho e logotipo  

image_path = 'assets/images/foto.png'
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

    texto_sem_base = re.sub(r"\[BASE DE CONHECIMENTO\].*", "", texto_limpo, flags=re.DOTALL).strip()

    # Converte \n\n e \n para quebras de linha reais
    texto_formatado = texto_sem_base.replace("\\n", "\n").replace("\n\n", "\n\n")

    
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




# Campo de entrada para a pergunta do usuário

# Interface do Streamlit

# Campo de entrada para a pergunta do usuário
#resposta_placeholder = st.empty()

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

#resposta_placeholder.container(height=300, border= False)
input = st.text_area(label="Faça uma pergunta:")
uploaded_file = st.file_uploader("Envie uma imagem para análise", type=["jpg", "jpeg", "png"])
#url = st.text_input(label= "Insira a url da imagem:")


# Botão para enviar a pergunta
if "generated_response" not in st.session_state:
    st.session_state.generated_response = None

if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = []
if st.button("Enviar"):
    
    if uploaded_file and input:
        image = Image.open(uploaded_file)

    # Redimensionar se maior que 1024px
        max_dim = max(image.size)
        if max_dim > 1024:
            resize_ratio = 1024 / max_dim
            new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.info(f"🔄 Imagem redimensionada para {new_size[0]}x{new_size[1]} para compatibilidade com a API.")
        else:
            st.success(f"✅ Tamanho da imagem: {image.size[0]}x{image.size[1]} (sem redimensionamento).")

        # Mostrar imagem na interface
        #st.image(image, caption="Imagem enviada", use_column_width=True)

        # Codificar imagem em base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        query = html.escape(input)
        response = responder(query, img_base64, st.session_state["session_id"], mensagens= st.session_state["messages"])
        
        st.session_state.user_input =""
        st.session_state["mensagens"].append({"role": "user", "content": input})
        st.session_state["mensagens"].append({"role": "assistant", "content": response}) 
      
        resposta = response
        st.session_state.generated_response = response
  
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

# configurações de layout - rodapé

st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #6c757d;
            text-align: center;
            padding: 10px;
            font-size: 0.8em;
            border-top: 1px solid #dee2e6;
        }
        </style>
        <div class="footer">
            Desenvolvido por Guilherme Felipe Breetz Rodovalho - AI-generated, for reference only
        </div>
        """,
        unsafe_allow_html=True
    )

