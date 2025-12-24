from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import AsyncChromiumLoader
from pypdf import PdfReader
from langchain_community.document_transformers import Html2TextTransformer
import json

load_dotenv()

#LO DE OS ES PARA UNA RUTA ABSOLUTA(PORQUE AQUI EL ARCHIVO ESTA EN /data)
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "guia-profesorado.pdf")
persist_db_path = os.path.join(current_dir, "chroma_db")

reader=PyPDFLoader(pdf_path)
#print(len(reader.pages))

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

docs=reader.load_and_split(text_splitter)
#print(docs[2])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#AQUI CREO LA BASE DE DATOS VECTORIAL Y SE GUARDA EN ESTA MISMA CARPETA
vector_store=Chroma(
    collection_name="guia_profesorado",
    embedding_function=embeddings,
    persist_directory=persist_db_path
)

# VERIFICACIÓN: Solo añadir si la colección está vacía
num_docs_actuales = vector_store._collection.count()
#_______________________________________________________________________________________________


#_______________________________________________________________________________________________



if num_docs_actuales == 0:
    print("La base de datos está vacía. Indexando documentos...")
    vector_store.add_documents(documents=docs)
    print("¡Indexación completada!")
else:
    print(f"La base de datos ya tiene {num_docs_actuales} documentos. No se añadieron duplicados.")

# Test rápido de búsqueda
query = "¿Qué dice sobre los horarios?"
docs_relacionados = vector_store.similarity_search(query, k=1)
if docs_relacionados:
    print("\nPrueba de búsqueda exitosa:")
    print(docs_relacionados[0].page_content[:150])
