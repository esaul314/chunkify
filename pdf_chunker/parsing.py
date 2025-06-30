import pdfplumber

def extract_text(filepath):
    with pdfplumber.open(filepath) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(filter(None, pages))

