import json
import io
from datetime import datetime

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

def create_pdf_bytes_plain_from_html(html, title="Itinerary Export"):
    # strip tags
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")  # join with newlines for <p>, <br>, etc.

    # then reuse your existing function or embed printing logic
    return create_pdf_bytes(text, title=title)

def create_pdf_bytes(text, title="Itinerary Export"):
    """
    Create a PDF in-memory from the provided text using reportlab and return bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    top = height - 0.75 * inch
    bottom_margin = 0.75 * inch

    # Basic text wrapping logic
    leading = 12  # line spacing
    max_width = width - left_margin - right_margin
    text_object = c.beginText()
    text_object.setTextOrigin(left_margin, top)
    text_object.setLeading(leading)
    text_object.setFont("Helvetica-Bold", 14)
    text_object.textLine(title)
    text_object.moveCursor(0, 6)
    text_object.setFont("Helvetica", 10)
    text_object.textLine("")  # blank line

    # split long lines
    for paragraph in text.split("\n"):
        # wrap manually
        words = paragraph.split(" ")
        line = ""
        for w in words:
            prospective = (line + " " + w).strip() if line else w
            # measure width via string width (approx)
            if c.stringWidth(prospective, "Helvetica", 10) <= max_width:
                line = prospective
            else:
                text_object.textLine(line)
                line = w
                # if we overflow page, add new page
                if text_object.getY() <= bottom_margin:
                    c.drawText(text_object)
                    c.showPage()
                    text_object = c.beginText()
                    text_object.setTextOrigin(left_margin, top)
                    text_object.setLeading(leading)
                    text_object.setFont("Helvetica", 10)
        # write remainder
        text_object.textLine(line)
        # small gap after each paragraph line if paragraph empty add an extra line
        # we've been adding line by line so it's fine

        if text_object.getY() <= bottom_margin:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText()
            text_object.setTextOrigin(left_margin, top)
            text_object.setLeading(leading)
            text_object.setFont("Helvetica", 10)

    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    return buffer.read()