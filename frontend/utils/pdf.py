from utils.request_api import extract_pdf_text


def extract_text_from_pdf(file):
    """Extracts text from the uploaded PDF file via the backend API."""
    return extract_pdf_text(file)
