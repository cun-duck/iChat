import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

# Tentukan direktori untuk menyimpan data NLTK (misalnya di folder lokal 'nltk_data')
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

# Tambahkan direktori tersebut ke nltk.data.path jika belum ada
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Pastikan resource 'punkt' dan 'punkt_tab' tersedia
for resource in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        # Resource name adalah bagian terakhir dari path, misalnya "punkt" atau "punkt_tab"
        resource_name = resource.split('/')[-1]
        nltk.download(resource_name, download_dir=nltk_data_dir)

# Inisialisasi model SentenceTransformer secara global
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, max_chunk_size=500, similarity_threshold=0.75):
    """
    Memecah teks menjadi chunk yang lebih kecil secara semantik.

    Args:
        text (str): Teks input yang akan di-chunk.
        max_chunk_size (int): Maksimal jumlah karakter per chunk.
        similarity_threshold (float): Threshold cosine similarity untuk menggabungkan kalimat.

    Returns:
        list: Daftar chunk teks.
    """
    # Pecah teks menjadi kalimat, secara eksplisit menggunakan bahasa 'english'
    sentences = sent_tokenize(text, language='english')
    
    # Dapatkan embedding untuk setiap kalimat sekaligus
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    chunks = []
    current_chunk = ""
    current_chunk_embedding = None
    current_chunk_sentence_count = 0

    for i, sentence in enumerate(sentences):
        sentence_embedding = embeddings[i]
        if not current_chunk:
            current_chunk = sentence
            current_chunk_sentence_count = 1
            current_chunk_embedding = sentence_embedding
        else:
            # Hitung cosine similarity antara embedding rata-rata chunk dan kalimat baru
            similarity = util.cos_sim(current_chunk_embedding, sentence_embedding).item()
            # Jika penambahan kalimat membuat chunk terlalu besar atau similarity di bawah threshold, buat chunk baru
            if len(current_chunk) + len(sentence) > max_chunk_size or similarity < similarity_threshold:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_chunk_sentence_count = 1
                current_chunk_embedding = sentence_embedding
            else:
                current_chunk += " " + sentence
                # Perbarui rata-rata embedding secara incremental
                current_chunk_embedding = (
                    current_chunk_embedding * current_chunk_sentence_count + sentence_embedding
                ) / (current_chunk_sentence_count + 1)
                current_chunk_sentence_count += 1

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
