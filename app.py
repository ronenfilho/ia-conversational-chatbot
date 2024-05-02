import streamlit as st
import os
import random

from groq import Groq
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


def main():
    """
    Esta função é o ponto de entrada principal da aplicação. 
    Ela configura o cliente Groq, a interface Streamlit e trata a interação do chat.
    Sempre responder em lingua portuguesa do Brasil. 
    """        

    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv()

    # Obtém o valor da variável de ambiente GROQ_API_KEY
    groq_api_key = os.getenv("GROQ_API_KEY")     

    # Exibe o logo do Groq
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    # O título e a mensagem de saudação da aplicação Streamlit
    st.title("Converse com o Groq!")
    st.write("Olá! Eu sou seu amigável chatbot Groq. \nPosso ajudar a responder suas perguntas, fornecer informações ou apenas conversar. Também sou super rápido! Vamos começar nossa conversa!")

    # Adiciona opções de personalização à barra lateral
    st.sidebar.title('Personalização')
    model = st.sidebar.selectbox(
        'Escolha um modelo',
        ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Comprimento da memória conversacional:', 1, 10, value = 5)

    memory=ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_input("Faça uma pergunta:")

    # Variável de estado da sessão
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input':message['human']},{'output':message['AI']})


    # Inicializa o objeto de chat Groq Langchain e a conversa
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

    conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )

    # Se o usuário fez uma pergunta,
    if user_question:
        
        # A resposta do chatbot é gerada enviando o prompt completo para a API Groq.
        response = conversation.invoke(user_question)
        message = {'human':user_question,'AI':response['response']}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response['response'])

if __name__ == "__main__":
    main()