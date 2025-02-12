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

# Sidebar untuk input pengguna
st.sidebar.title("Pengaturan")

# Input HF_TOKEN
hf_token = st.sidebar.text_input("Masukkan Hugging Face Token:", type="password")

# Upload file PDF
uploaded_file = st.sidebar.file_uploader("Unggah file PDF", type=["pdf"])

# Prompt optimizer
instructions = st.sidebar.text_area("Tambahkan instruksi untuk prompt optimizer (opsional):")

# Inisialisasi model AI jika HF_TOKEN tersedia
if hf_token:
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token
    )
else:
    st.sidebar.warning("Silakan masukkan Hugging Face Token untuk menggunakan model.")

# Proses file PDF jika diunggah
chunks = []
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    st.sidebar.write(f"File berhasil diunggah dan dibagi menjadi {len(chunks)} chunk.")

# Fungsi untuk menghasilkan respons dari model
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

# Halaman utama
st.title("Chatbot LLM dengan RAG Lokal")

# Input pertanyaan
user_input = st.text_input("Masukkan pertanyaan Anda:")

# Tombol kirim
if st.button("Kirim"):
    if not hf_token:
        st.error("Silakan masukkan Hugging Face Token terlebih dahulu.")
    elif not user_input:
        st.error("Silakan masukkan pertanyaan Anda.")
    else:
        if instructions:
            optimized_prompt = optimize_prompt(user_input, instructions)
            response, token_usage = generate_response(optimized_prompt)
        else:
            response, token_usage = generate_response(user_input)

        if uploaded_file:
            relevant_chunk = retrieve_relevant_chunk(user_input, chunks)
            st.write("Jawaban:", response)
            st.write("Chunk yang digunakan:", relevant_chunk)
        else:
            st.write("Jawaban:", response)
        
        # Feedback token usage
        st.write(f"Jumlah token yang digunakan: {token_usage}")
