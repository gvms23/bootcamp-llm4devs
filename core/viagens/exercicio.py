import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=api_key)

def ask_bot(message, history):
    history_template = [
        ("system", "você é um agente de viagens, trabalhe como um agente de viagens, não responda nada que não esteja nesse contexto de agência de viagens"),
        ("system", "Você é um bot que deve agir como um agente de viagens"),
        ("system", "Quando souber o nome do usuário em cada resposta procure trazer isso."),
        ("system", "Responda em no máximo 20 palavras."),
        ("human", "{user_input}"),
        ("system", "Responda em pt-br."),
        ("system", "Quando a pergunta não tiver ligação com viagens, mas amplie a visão, então tudo ligado a viagens como tempo, clima, temperatura, distância, as perguntas ligadas a isso responda também, caso o assunto seja muito distante disso, responda: 'Sou um agente treinado para ser seu agente de viagens, posso responder apenas nesse contexto.'."),
        ("system", "leve em consideração o histórico a seguir:")
    ]
    for item in history:
        history_template.append(("system", item))

    chat_template = ChatPromptTemplate.from_messages(history_template)
    chain = chat_template | llm
    resposta = chain.invoke({"user_input": message})
    print(resposta.response_metadata.token_usage)
    return resposta.content

chat = gr.ChatInterface(
        ask_bot,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Pergunte algo sobre para o SuperBot!", container=False, scale=7),
        title="SuperBot",
        description="Nosso SuperBot está pronto para atender suas perguntas!",
        theme="soft",
        examples=["Quem é você?", "O que você sabe sobre futebol?", "O que é uma abelha?"],
        cache_examples=True,
        retry_btn=None,
        undo_btn=None,
        submit_btn="Enviar",
        clear_btn="Limpar",
    )

chat.launch(share=True, server_name="0.0.0.0", server_port=7860)