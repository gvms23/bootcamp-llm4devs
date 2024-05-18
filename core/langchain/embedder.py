from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
load_dotenv()
# se começar a usar um tipo de modelo de embedding, NÃO DÁ PRA USAR OUTRO
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))

def embed_doc(doc):
    return embeddings.embed_query(doc)

def embed_docs(docs):
    return embeddings.embed_documents(docs)