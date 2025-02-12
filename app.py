import streamlit as st
from huggingface_hub import InferenceClient
from utils.pdf_processor import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.rag import retrieve_relevant_chunk
from utils.prompt_optimizer import optimize_prompt
import time

# Muat CSS kustom
def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar untuk input dan feedback
st.sidebar.title("Settings")
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
instructions = st.sidebar.text_area("Add instructions for prompt optimization (optional):")
st.sidebar.subheader("Feedback")
token_feedback = st.sidebar.empty()   # Tempat feedback token
chunk_feedback = st.sidebar.empty()   # Tempat feedback chunk yang digunakan

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Default context dan prompt
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

# Fungsi untuk mereset data pengguna (jika terjadi inaktivitas)
def reset_user_data():
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.uploaded_file = None
    st.sidebar.success("User data has been cleared due to inactivity.")

# Cek inaktivitas (lebih dari 2 menit)
current_time = time.time()
if current_time - st.session_state.last_activity > 120:
    reset_user_data()
st.session_state.last_activity = current_time

# Proses file PDF yang diunggah
if uploaded_file:
    try:
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.chunks = chunk_text(text)
        st.sidebar.success(f"PDF successfully processed! File split into {len(st.session_state.chunks)} chunks.")
        st.toast(f"File split into {len(st.session_state.chunks)} chunks.")
    except ValueError as e:
        st.sidebar.error(str(e))

# Inisialisasi model AI jika HF_TOKEN sudah dimasukkan
if hf_token:
    client = InferenceClient(provider="hf-inference", api_key=hf_token)
else:
    st.sidebar.warning("Please enter your Hugging Face Token to use the model.")

# Fungsi untuk menghasilkan respons dari model
def generate_response(prompt, context=None):
    if not hf_token:
        st.error("Please enter your Hugging Face Token first.")
        return "", 0

    messages = [{"role": "user", "content": prompt}]
    if context:
        messages.insert(0, {"role": "system", "content": context})
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=messages,
        max_tokens=1500
    )
    return completion.choices[0].message.content, completion.usage.total_tokens

# Fungsi untuk menampilkan riwayat chat dengan tampilan HTML kustom
def display_chat_history():
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
        # Tampilkan informasi chunk (jika ada) untuk pesan asisten
        if role == "assistant" and "relevant_chunk" in message:
            st.markdown(
                f"""
                <div style="font-size: 8px; color: #08f700;">
                    Relevant chunk used: {message["relevant_chunk"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

# Tampilkan chat history yang sudah tersimpan
display_chat_history()

# Input pertanyaan pengguna menggunakan st.chat_input agar terlihat natural
if user_input := st.chat_input("Type your question here..."):
    # Tambahkan pesan pengguna ke riwayat chat dan tampilkan
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    st.markdown(
        f"""
        <div class="chat-container">
            <div class="user-bubble">
                <div class="user-avatar"></div>
                {user_input}
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Placeholder untuk streaming respons asisten
    assistant_placeholder = st.empty()
    response_stream = ""
    
    # Dapatkan respons penuh dari model (dengan atau tanpa chunk)
    if st.session_state.chunks:
        relevant_chunk = retrieve_relevant_chunk(user_input, st.session_state.chunks)
        full_response, token_usage = generate_response(user_input, context=relevant_chunk)
        token_feedback.markdown(f"**Tokens used:** {token_usage}")
        chunk_feedback.markdown(f"**Relevant chunk used:** {relevant_chunk}")
    else:
        full_response, token_usage = generate_response(user_input, context=default_context)
        token_feedback.markdown(f"**Tokens used:** {token_usage}")
        chunk_feedback.markdown("**No chunk used (no PDF uploaded).**")
    
    # Streaming: tampilkan respons kata demi kata secara real-time
    for word in full_response.split():
        response_stream += word + " "
        assistant_placeholder.markdown(
            f"""
            <div class="chat-container">
                <div class="assistant-bubble">
                    <div class="assistant-avatar"></div>
                    {response_stream}
                    <div class="timestamp">{time.strftime("%H:%M")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.05)  # jeda singkat untuk efek streaming

    # Setelah streaming selesai, simpan pesan asisten ke riwayat chat
    timestamp = time.strftime("%H:%M")
    message_data = {
        "role": "assistant",
        "content": full_response,
        "timestamp": timestamp
    }
    if st.session_state.chunks:
        message_data["relevant_chunk"] = relevant_chunk
    st.session_state.messages.append(message_data)
