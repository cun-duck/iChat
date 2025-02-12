def chunk_text(text, chunk_size=500):
    """
    Splits text into smaller chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
