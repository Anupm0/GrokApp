import streamlit as st
import os
from groq import Groq
import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ['groq_api_key']

def main():
    st.title("Groq Chat App")
    st.sidebar.title("Select an LLM")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['Mixtral-8x7b-32768', 'llama2-70b-4096']
    )

    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
    memory = ConversationBufferMemory(k=conversational_memory_length)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.write("Chat History")
    for message in st.session_state.chat_history:
        st.markdown(f"""
        <div style="margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #f0f0f5; color: black;">
            <p><strong>You:</strong> {message['human']}</p>
            <p><strong>Chatbot:</strong> {message['AI']}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.form("my_form"):
        user_question = st.text_area("Ask a question....")
        submitted = st.form_submit_button("Enter")

    # Update memory with previous chat history
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    if submitted:
        response = conversation(user_question)
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)
        st.markdown(f"""
        <div style="margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #e0ffe0; color: black;">
            <p><strong>You:</strong> {user_question}</p>
            <p><strong>Chatbot:</strong> {response['response']}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
