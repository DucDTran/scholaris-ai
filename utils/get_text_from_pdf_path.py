import fitz

def get_text_from_pdf_path(pdf_path):
    """Extracts text from a PDF file path using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text