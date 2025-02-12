import PyPDF2

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.

    Args:
        file (UploadedFile): The uploaded PDF file from Streamlit.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        # Membaca file PDF menggunakan PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = ""

        # Iterasi melalui setiap halaman dan ekstrak teks
        for page in reader.pages:
            text += page.extract_text()

        # Jika tidak ada teks yang diekstrak, berikan peringatan
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF. Please ensure the file contains readable text.")

        return text
    except Exception as e:
        # Tangani kesalahan jika terjadi masalah saat membaca PDF
        raise ValueError(f"An error occurred while processing the PDF: {str(e)}")
