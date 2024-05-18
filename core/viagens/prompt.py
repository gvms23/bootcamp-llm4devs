from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=api_key)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um superbot especializado como um guia turístico de pindamonhangaba"),
        ("human", "{user_input}"),
        ("system", "Responda em no máximo 50 palavras."),
        ("system", "Responda em pt-br.")
    ]
)

chain = chat_template | llm

resposta = chain.invoke({"user_input":"Você recomenda ir para Recife ou São Paulo"})
print(resposta)