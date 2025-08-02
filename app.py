# import streamlit as st
# from groq_rag import generate_answer

# st.set_page_config(page_title="PDF Chatbot - Adam & Eve Inc", layout="centered")
# st.title("📄 Chat with Adam & Eve Inc PDFs")

# query = st.text_input("Ask a question based on the uploaded PDFs:")

# if st.button("Ask") and query:
#     with st.spinner("Thinking..."):
#         answer = generate_answer(query)
#         st.success(answer)



import streamlit as st
from groq_rag import generate_answer

# Page setup
st.set_page_config(page_title="📄 Adam & Eve PDF Chatbot", layout="centered")

# Custom CSS to fix input at the bottom and allow scrolling chat
st.markdown("""
    <style>
        .chat-scroll {
            max-height: 75vh;
            overflow-y: auto;
            padding-bottom: 10px;
            display: flex;
            flex-direction: column;
        }
        .chat-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            z-index: 100;
            border-top: 1px solid #ccc;
        }
        .block-container {
            padding-bottom: 100px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and instructions
st.title("🤖 Chat with Adam & Eve Inc Assistant")
st.markdown("")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input (must come early to be processed first)
user_query = st.chat_input("Type your question here...")

# ✅ Immediately handle new user input (this fixes the delay bug)
if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.spinner("Searching the documents..."):
        answer = generate_answer(user_query)
    st.session_state.chat_history.append(("assistant", answer))

# ✅ Display messages only if chat history exists
if st.session_state.chat_history:
    with st.container():
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                if role == "user":
                    st.markdown(f"🧑‍💼 **You:** {message}", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"""
                        <div style="background-color:#f9f9f9;padding:15px;border-radius:10px;border:1px solid #e0e0e0">
                        <strong>🤖 Assistant:</strong><br>{message.replace('\n', '<br>')}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.markdown('</div>', unsafe_allow_html=True)


