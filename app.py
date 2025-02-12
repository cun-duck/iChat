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

# Sidebar for user inputs
st.sidebar.title("Settings")

# Input HF_TOKEN
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password")

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Prompt optimizer
instructions = st.sidebar.text_area("Add instructions for prompt optimization (optional):")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
if "chunks" not in st.session_state:
    st.session_state.chunks = []

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
        st.sidebar.success("PDF successfully processed!")
        st.sidebar.write(f"File split into {len(st.session_state.chunks)} chunks.")
    except ValueError as e:
        st.sidebar.error(str(e))

# Function to generate response from the model
def generate_response(prompt, context=None):
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

# User input for questions
if user_input := st.chat_input("Type your question here..."):
    # Add user message to chat history
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": timestamp})

    # Generate response
    if not hf_token:
        st.error("Please enter your Hugging Face Token first.")
    else:
        if instructions:
            optimized_prompt = optimize_prompt(user_input, instructions)
            response, token_usage = generate_response(optimized_prompt)
        else:
            response, token_usage = generate_response(user_input)

        if st.session_state.chunks:
            # Retrieve relevant chunk using RAG
            relevant_chunk = retrieve_relevant_chunk(user_input, st.session_state.chunks)
            full_response = f"**Answer:** {response}\n\n**Relevant chunk used:** {relevant_chunk}"
        else:
            full_response = f"**Answer:** {response}"

        # Add assistant message to chat history
        timestamp = time.strftime("%H:%M")
        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": timestamp})

        # Feedback on token usage
        st.write(f"Tokens used: {token_usage}")
        st.progress(token_usage / 1500)
