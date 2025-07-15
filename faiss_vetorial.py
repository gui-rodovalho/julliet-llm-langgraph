from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os

VECTOR_INDEX_PATH = "vector_index"


def save_to_faiss(content: str):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    vectors = FAISS.from_texts(content, embedding=embeddings)

    if os.path.exists(VECTOR_INDEX_PATH):
        vectorstore = FAISS.load_local(VECTOR_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.merge_from(vectors)
       # vectorstore.add_texts([content])
        vectorstore.save_local(VECTOR_INDEX_PATH)
    else:
        
        
        vectors.save_local(VECTOR_INDEX_PATH)

   