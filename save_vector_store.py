from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import traceback
import os      

   
def add_to_faiss(save_dir, save_path):
    print("chegou no add to faiss")
    index_path = "rag_index" 

    # trasforma os documentos enviados para o rag em um indice de vetores

    caminho_arquivo = os.path.join(save_dir, save_path)
    if os.path.isfile(caminho_arquivo) and caminho_arquivo.lower().endswith('.pdf'):
            try:
                pdf_loader = PyPDFLoader(caminho_arquivo)
                pdf_carregado = pdf_loader.load()
                doc = pdf_carregado                
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(doc)
                embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
                vectors = FAISS.from_documents(final_documents, embedding=embeddings)
                faiss_index = vectors.index
                print(f"Total de vetores no índice: {faiss_index.ntotal}")
                
                if not os.path.exists(index_path):
                    vectors.save_local(index_path)

                    # index_name deve ser sempre = index
                    
                    print('\n NÃO ENCONTROU FAISS.INDEX\n')
                else:
                    faiss_store = FAISS.load_local(index_path,embeddings= embeddings, allow_dangerous_deserialization=True)
                    faiss_store.merge_from(vectors)
                    index = faiss_store.index
                    print(f"Total de vetores no merge: {index.ntotal}")
                    faiss_store.save_local(folder_path= '.venv/data/faiss_index',index_name= "index")
            
                print(f"arquivo {caminho_arquivo} salvo com sucesso - FAISS")
                retorno = f"arquivo {caminho_arquivo} salvo com sucesso"

                


            except Exception as e:
                print(f"Erro ao carregar o pdf {caminho_arquivo}: {e}")
                traceback.print_exc()
                retorno = f"Erro ao carregar o pdf {caminho_arquivo}: {e}"
    return retorno