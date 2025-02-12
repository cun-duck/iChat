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

# Sidebar untuk input pengguna dan feedback
st.sidebar.title("Settings")

# Input HF_TOKEN
hf_token = st.sidebar.text_input("Enter Hugging Face Token:", type="password")

# Upload PDF file
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Prompt optimizer (opsional)
instructions = st.sidebar.text_area("Add instructions for prompt optimization (optional):")

# Feedback section di sidebar
st.sidebar.subheader("Feedback")
token_feedback = st.sidebar.empty()  # Placeholder untuk informasi penggunaan token
chunk_feedback = st.sidebar.empty()  # Placeholder untuk informasi chunk yang digunakan

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

# Fungsi untuk mereset data pengguna jika terjadi inaktivitas
def reset_user_data():
    st.session_state.messages = []
    st.session_state.chunks = []
    st.session_state.uploaded_file = None
    st.sidebar.success("User data has been cleared due to inactivity.")

# Cek inaktivitas (jika lebih dari 2 menit)
current_time = time.time()
if current_time - st.session_state.last_activity > 120:  # 120 detik = 2 menit
    reset_user_data()

# Update waktu aktivitas terakhir
st.session_state.last_activity = current_time

# Proses file PDF yang diunggah
if uploaded_file:
    try:
        # Ekstrak teks dari PDF
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.chunks = chunk_text(text)
        st.sidebar.success(f"PDF successfully processed! File split into {len(st.session_state.chunks)} chunks.")
        st.toast(f"File split into {len(st.session_state.chunks)} chunks.")  # Notifikasi pop-up
    except ValueError as e:
        st.sidebar.error(str(e))

# Inisialisasi model AI jika HF_TOKEN telah dimasukkan
if hf_token:
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token
    )
else:
    st.sidebar.warning("Please enter your Hugging Face Token to use the model.")

# Fungsi untuk menghasilkan respons dari model
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

# Tampilan Halaman Utama
st.title("LLM Chatbot with Local RAG")

# Menampilkan riwayat chat yang tersimpan
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    if role == "user":
        st.chat_message("user").write(f"{content} ({timestamp})")
    else:
        assistant_msg = st.chat_message("assistant")
        assistant_msg.write(f"{content} ({timestamp})")
        # Jika ada informasi chunk yang relevan, tampilkan juga
        if "relevant_chunk" in message:
            st.write(f"<div style='font-size: 10px; color: #888;'>Relevant chunk used: {message['relevant_chunk']}</div>", unsafe_allow_html=True)

# Input pertanyaan pengguna dengan st.chat_input
if user_input := st.chat_input("Type your question here..."):
    # Tambahkan pesan pengguna ke riwayat chat dan tampilkan
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": timestamp})
    st.chat_message("user").write(user_input)
    
    # Placeholder untuk respons asisten dengan streaming
    with st.chat_message("assistant") as assistant_msg:
        placeholder = st.empty()  # Placeholder untuk update streaming respons
        
        # Jika ada PDF yang telah diproses, cari chunk yang relevan
        if st.session_state.chunks:
            relevant_chunk = retrieve_relevant_chunk(user_input, st.session_state.chunks)
            full_response, token_usage = generate_response(user_input, context=relevant_chunk)
            token_feedback.markdown(f"**Tokens used:** {token_usage}")
            chunk_feedback.markdown(f"**Relevant chunk used:** {relevant_chunk}")
        else:
            full_response, token_usage = generate_response(user_input, context=default_context)
            token_feedback.markdown(f"**Tokens used:** {token_usage}")
            chunk_feedback.markdown("**No chunk used (no PDF uploaded).**")
        
        # Simulasi streaming: tampilkan respons secara bertahap (misalnya, per kata)
        response_stream = ""
        for word in full_response.split():
            response_stream += word + " "
            placeholder.write(response_stream)
            time.sleep(0.05)  # jeda pendek untuk efek streaming

        # Setelah streaming selesai, simpan respons penuh ke riwayat chat
        timestamp = time.strftime("%H:%M")
        message_data = {
            "role": "assistant",
            "content": full_response,
            "timestamp": timestamp
        }
        if st.session_state.chunks:
            message_data["relevant_chunk"] = relevant_chunk
        st.session_state.messages.append(message_data)
