import time
from os import environ as env

import streamlit as st
from dotenv import load_dotenv
from ResumeChatAPI import ResumeChatAPI

# Load the environment
load_dotenv()

# Set Wide Config
st.set_page_config(layout='wide')

# Users AI Assistant
st.header(f"{env.get('USER_NAME')} AI Assistant", divider='rainbow')
st.markdown(f"<a href={env.get('RESUME_LINK')} style='text-decoration: none;'>Resume 🔗</a>", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ResumeChatAPI" not in st.session_state:
    st.session_state.ResumeChatAPI = ResumeChatAPI()

# Display chat messages from history on app rerun
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    history = []
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response, history = st.session_state.ResumeChatAPI.get_answers(prompt, history)

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
