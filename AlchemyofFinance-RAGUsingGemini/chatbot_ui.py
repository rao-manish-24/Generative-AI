import streamlit as st
import os
import shelve
import requests
from chatbot_backend import Model

st.title("George Soros Bot 🤖")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])
        # return []

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Input for API URL
# api_url = st.text_input("Enter the GET API URL:", "http://127.0.0.1:5000/response")
api_url = "http://127.0.0.1:5000/ask_response"

def response_api(user_prompt):
    try:
        print("user prompt is: ", user_prompt)
        # Make a GET request to the API with user_prompt as a parameter
        response = requests.get(api_url, params={"prompt": user_prompt})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Parse JSON response
            st.success("Data fetched successfully!")
            st.json(data)  # Display the data as formatted JSON
        else:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Exception occurred in API call"


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []  # This is custom state that we're using to store the chat history, the word messages is just a name we gave it, it can be anything
        save_chat_history([])

print(st.session_state) # prints the current session state


# st.session_state.messages.append({"role": "assistant", "content": "How are you ?"})
# st.session_state.messages.append({"role": "user", "content": "I am fine , thanks"})

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    print("prompt " ,prompt)
    print("prompt type : ", type(prompt))

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        print('--------------------------------------')

        bot_response = response_api(prompt)['response']
        print('type of bot response :', type(bot_response))
        print(bot_response)
        print('***************************************')
        # full_response = model.get_model_response(prompt)

        message_placeholder.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})



    