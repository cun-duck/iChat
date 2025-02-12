import streamlit as st
from huggingface_hub import InferenceClient
from utils.pdf_processor import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.rag import retrieve_relevant_chunk
from utils.prompt_optimizer import optimize_prompt
import time

# Load CSS
def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar for user inputs and feedback
st.sidebar.title("Settings")

# Input HF_TOKEN
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password")

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Prompt optimizer
instructions = st.sidebar.text_area("Add instructions for prompt optimization (optional):")

# Feedback section in the sidebar
st.sidebar.subheader("Feedback")
token_feedback = st.sidebar.empty()  # Placeholder for token usage
chunk_feedback = st.sidebar.empty()  # Placeholder for chunk used

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Default context and prompt
default_context = """
This is a general-purpose chatbot that can answer questions about technology, programming, AI, and other topics.
For example:
- What is machine learning?
- How does a neural network work?
- Explain the concept of natural language processing.
"""
default_prompt = """
You are a helpful assistant. If no specific context is provided, answer general questions based on your training data.
If the question is unclear or cannot be answered, politely inform the user.
"""

# Function to reset user data
def reset_user_data():
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.uploaded_file = None
    st.sidebar.success("User data has been cleared due to inactivity.")

# Check for inactivity
current_time = time.time()
if current_time - st.session_state.last_activity > 120:  # 120 seconds = 2 minutes
    reset_user_data()

# Update last activity time whenever there is user interaction
st.session_state.last_activity = current_time

# Process PDF file if uploaded
if uploaded_file:
    try:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.chunks = chunk_text(text)
        st.sidebar.success(f"PDF successfully processed! File split into {len(st.session_state.chunks)} chunks.")
        st.toast(f"File split into {len(st.session_state.chunks)} chunks.")  # Pop-up notification
    except ValueError as e:
        st.sidebar.error(str(e))

# Initialize AI model if HF_TOKEN is provided
if hf_token:
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token
    )
else:
    st.sidebar.warning("Please enter your Hugging Face Token to use the model.")

# Function to generate response from the model
def generate_response(prompt, context=None):
    if not hf_token:
        st.error("Please enter your Hugging Face Token first.")
        return "", 0

    messages = [
        {"role": "user", "content": prompt}
    ]
    if context:
        messages.insert(0, {"role": "system", "content": context})
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=messages,
        max_tokens=1500
    )
    return completion.choices[0].message.content, completion.usage.total_tokens

# Main page
st.title("LLM Chatbot with Local RAG")

# Display chat messages from history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    avatar_class = "user-avatar" if role == "user" else "assistant-avatar"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"

    st.markdown(
        f"""
        <div class="chat-container">
            <div class="{bubble_class}">
                <div class="{avatar_class}"></div>
                {content}
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show relevant chunk (if any) below assistant's response
    if role == "assistant" and "relevant_chunk" in message:
        st.markdown(
            f"""
            <div style="font-size: 5px; color: #3deb2a;">
                Relevant chunk used: {message["relevant_chunk"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

# User input for questions
if user_input := st.chat_input("Type your question here..."):
    # Add user message to chat history
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": timestamp})

    # Generate response with spinner
    with st.spinner("Generating response..."):
        if st.session_state.chunks:
            relevant_chunk = retrieve_relevant_chunk(user_input, st.session_state.chunks)
            full_response, token_usage = generate_response(user_input, context=relevant_chunk)

            # Update feedback in the sidebar
            token_feedback.markdown(f"**Tokens used:** {token_usage}")
            chunk_feedback.markdown(f"**Relevant chunk used:** {relevant_chunk}")

            # Add assistant message to chat history with relevant chunk
            timestamp = time.strftime("%H:%M")
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": timestamp,
                "relevant_chunk": relevant_chunk
            })
        else:
            full_response, token_usage = generate_response(user_input, context=default_context)

            # Update feedback in the sidebar
            token_feedback.markdown(f"**Tokens used:** {token_usage}")
            chunk_feedback.markdown("**No chunk used (no PDF uploaded).**")

            # Add assistant message to chat history without relevant chunk
            timestamp = time.strftime("%H:%M")
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": timestamp
            })
