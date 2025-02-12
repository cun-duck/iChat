from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunk(query, chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + chunks)
    similarities = cosine_similarity(vectors[0], vectors[1:])
    most_similar_index = similarities.argmax()
    return chunks[most_similar_index]
