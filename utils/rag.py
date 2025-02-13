from sentence_transformers import SentenceTransformer, util

# Inisialisasi model sekali saja
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunk(query, chunks, precomputed_chunk_embeddings=None):
    """
    Mengambil chunk yang paling relevan berdasarkan query menggunakan cosine similarity.
    
    Args:
        query (str): Pertanyaan atau query pengguna.
        chunks (list): List string dari chunk teks.
        precomputed_chunk_embeddings (Tensor, optional): Embedding yang sudah dihitung untuk chunks.
    
    Returns:
        str: Chunk yang paling relevan.
    """
    # Encode query menjadi tensor
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Gunakan embedding yang sudah dihitung jika tersedia, agar tidak menghitung ulang
    if precomputed_chunk_embeddings is None:
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    else:
        chunk_embeddings = precomputed_chunk_embeddings
    
    # Hitung cosine similarity antara query dan setiap chunk
    similarities = util.cos_sim(query_embedding, chunk_embeddings)
    
    # Dapatkan indeks chunk dengan similarity tertinggi
    most_similar_index = similarities.argmax().item()
    return chunks[most_similar_index]
