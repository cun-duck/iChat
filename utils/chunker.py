import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

# Periksa apakah resource 'punkt' sudah tersedia, jika tidak, download resource tersebut.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Inisialisasi model secara global untuk efisiensi
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
    # Pecah teks menjadi kalimat
    sentences = sent_tokenize(text)
    
    # Dapatkan embedding untuk setiap kalimat secara sekaligus
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    chunks = []
    current_chunk = ""
    current_chunk_embedding = None
    current_chunk_sentence_count = 0

    for i, sentence in enumerate(sentences):
        sentence_embedding = embeddings[i]
        # Jika chunk saat ini kosong, mulai dengan kalimat ini
        if not current_chunk:
            current_chunk = sentence
            current_chunk_sentence_count = 1
            current_chunk_embedding = sentence_embedding
        else:
            # Hitung cosine similarity antara rata-rata embedding chunk saat ini dengan embedding kalimat baru
            similarity = util.cos_sim(current_chunk_embedding, sentence_embedding).item()
            # Jika menambahkan kalimat melebihi batas ukuran atau similarity di bawah threshold, mulai chunk baru
            if len(current_chunk) + len(sentence) > max_chunk_size or similarity < similarity_threshold:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_chunk_sentence_count = 1
                current_chunk_embedding = sentence_embedding
            else:
                # Gabungkan kalimat ke chunk yang ada
                current_chunk += " " + sentence
                # Perbarui rata-rata embedding chunk secara incremental
                current_chunk_embedding = (current_chunk_embedding * current_chunk_sentence_count + sentence_embedding) / (current_chunk_sentence_count + 1)
                current_chunk_sentence_count += 1

    # Tambahkan chunk terakhir jika ada
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
