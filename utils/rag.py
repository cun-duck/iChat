from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunk(query, chunks):
    query_embedding = model.encode(query)
    chunk_embeddings = model.encode(chunks)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)
    most_similar_index = similarities.argmax().item()
    return chunks[most_similar_index]
