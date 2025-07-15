# Copyright 2024 Guilherme Rodovalho
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


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import os



def get_retriever(path):
    # recupera os vetores para alimentar o contexto da LLM
    
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    print("\n CHEGOU NO ELSE PARA LER O BANCO LOCAL \n")
    vectors = FAISS.load_local(path, embeddings= embeddings,allow_dangerous_deserialization= True) 
    vetores = faiss.read_index(f"{path}/index.faiss")
    vetores_indice = vetores.ntotal
    print(f"Total de vetores no índice: {vetores_indice}")
    retriever = vectors
        
    return retriever, vetores_indice

def get_relevant_documents(input, path):
    
    retriever, vetores_indice = get_retriever(path)

    if vetores_indice < 5: # é necessário analisar a partir de quantos vetores já se torna possível efetuar busca por mmr
        doc_relevant = retriever
        print(f"\n\nesse é o recuperador \n\n{doc_relevant}")
        return doc_relevant
    
    else:
        
        #pesquisar dentro dos vetores para retornar documentos mais importantes
        vectors = retriever.as_retriever( search_type="mmr", search_kwargs={'k': 3, "lambda_mult": 0.85})
        #alterar para buscar melhores resultados
        recuperador = vectors.invoke(input)
       # print(f"\n\nesse é o recuperador \n\n{recuperador}")
        #embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
        #vectors = FAISS.from_documents(recuperador, embedding=embeddings)
        #doc_relevant = vectors.as_retriever()
        return recuperador
    
   