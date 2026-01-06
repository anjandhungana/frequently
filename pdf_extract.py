# pdf_extract.py

import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Best-effort PDF text extraction.
    Tries pdfplumber first, then PyPDF2.
    Returns extracted text (may be empty if scanned PDF/Images).
    """
    text_parts = []

    # pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
    except Exception:
        pass

    # fallback: PyPDF2
    if not text_parts:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        except Exception:
            pass

    return "\n".join(text_parts).strip()