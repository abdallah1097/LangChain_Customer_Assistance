# frontend.py
import streamlit as st
import requests

def main():
    st.title("Restaurant Chatbot")
    
    st.write(
        """
        <style>
            body {
                background-image: url('https://i.redd.it/qwd83nc4xxf41.jpg');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                padding: 20px;
            }
            .container {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
                gap: 20px;
            }
            .input-container {
                display: flex;
                gap: 10px;
            }
            .error-message {
                background-color: #FFCCCC;
                color: red;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    chat_history = st.session_state.get('chat_history', [])
    
    st.write('<div style="display: flex; justify-content: center; align-items: center;  flex-direction: column; gap: 20px;">', unsafe_allow_html=True)
    
    st.write('<div style="display: flex; gap: 10px;">', unsafe_allow_html=True)
    user_input = st.text_input("Your Assistance", key="user_input", placeholder="Ask me anything...")
    st.write('</div>', unsafe_allow_html=True)
    
    st.write('<div style="display: flex; gap: 10px;">', unsafe_allow_html=True)
    button_clicked = st.button("Send")
    st.write('</div>', unsafe_allow_html=True)
    
    if button_clicked:
        if user_input.strip() == "":
            st.write('<div class="error-message">Please enter your question!</div>', unsafe_allow_html=True)
        else:
            chat_history.append(("user", user_input))
            
            response = requests.post("http://localhost:5000/query", json={"user_input": user_input})
            chatbot_response = response.json().get("response")
            chat_history.append(("chatbot", chatbot_response))
            
            st.session_state.chat_history = chat_history
    
    st.write('</div>', unsafe_allow_html=True)
    
    display_chat_history(chat_history)

def display_chat_history(chat_history):
    for sender, message in chat_history:
        if sender == "user":
            st.write(f'<div style="background-color: #DCF8C6; color: black; padding: 12px; border-radius: 20px; text-align: right; margin-bottom: 12px;">{message}</div>', unsafe_allow_html=True)
        else:
            st.write(f'<div style="background-color: #E5E5EA; color: black;  padding: 12px; border-radius: 20px; text-align: left; margin-bottom: 12px;">{message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
