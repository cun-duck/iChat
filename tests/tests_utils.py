import unittest
from utils.pdf_processor import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.rag import retrieve_relevant_chunk
from utils.prompt_optimizer import optimize_prompt

class TestUtils(unittest.TestCase):
    def test_extract_text_from_pdf(self):
        # Mock PDF file
        pass

    def test_chunk_text(self):
        text = "This is a sample text."
        chunks = chunk_text(text, chunk_size=5)
        self.assertEqual(len(chunks), 4)

    def test_retrieve_relevant_chunk(self):
        query = "sample"
        chunks = ["This is a", "sample text.", "Another chunk."]
        relevant_chunk = retrieve_relevant_chunk(query, chunks)
        self.assertEqual(relevant_chunk, "sample text.")

    def test_optimize_prompt(self):
        prompt = "What is the capital of France?"
        instructions = "Provide a concise answer."
        optimized_prompt = optimize_prompt(prompt, instructions)
        self.assertIn(instructions, optimized_prompt)

if __name__ == "__main__":
    unittest.main()
