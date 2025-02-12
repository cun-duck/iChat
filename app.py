import streamlit as st
from huggingface_hub import InferenceClient
from utils.pdf_processor import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.rag import retrieve_relevant_chunk
from utils.prompt_optimizer import optimize_prompt

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

# Initialize AI model if HF_TOKEN is provided
if hf_token:
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token
    )
else:
    st.sidebar.warning("Please enter your Hugging Face Token to use the model.")

# Process PDF file if uploaded
chunks = []
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    st.sidebar.write(f"File uploaded successfully and split into {len(chunks)} chunks.")

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="user-bubble">
                    {message["content"]}
                    <div class="timestamp">{message.get("timestamp", "")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif message["role"] == "assistant":
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="assistant-bubble">
                    {message["content"]}
                    <div class="timestamp">{message.get("timestamp", "")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# User input for questions
if user_input := st.chat_input("Type your question here..."):
    # Add user message to chat history
    import time
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

        if uploaded_file:
            relevant_chunk = retrieve_relevant_chunk(user_input, chunks)
            full_response = f"**Answer:** {response}\n\n**Relevant chunk used:** {relevant_chunk}"
        else:
            full_response = f"**Answer:** {response}"

        # Add assistant message to chat history
        timestamp = time.strftime("%H:%M")
        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": timestamp})
