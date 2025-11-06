import json
import io
from datetime import datetime

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, Color
import io
import re


def create_pdf_bytes_plain_from_html(html, title="Itinerary Export"):
    # strip tags
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")  # join with newlines for <p>, <br>, etc.

    # then reuse your existing function or embed printing logic
    return create_pdf_bytes_flowable_styled(text, title=title)


def create_pdf_bytes_flowable_styled(text, title="Itinerary Export"):
    buffer = io.BytesIO()

    # 1. Setup DocTemplate and Styles
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    # Define Custom Styles for hierarchy
    styles.add(ParagraphStyle(name='DayHeader',
                              parent=styles['Heading2'],
                              fontSize=14,
                              fontName='Helvetica-Bold',
                              spaceBefore=14,
                              spaceAfter=6,
                              textColor=Color(0.2, 0.2, 0.5)))  # Dark Blue for Day

    styles.add(ParagraphStyle(name='ActivityHeader',
                              parent=styles['Normal'],
                              fontSize=11,
                              fontName='Helvetica-Bold',
                              spaceBefore=8,
                              spaceAfter=4))

    styles.add(ParagraphStyle(name='DetailText',
                              parent=styles['Normal'],
                              fontSize=9,
                              spaceBefore=1,
                              spaceAfter=1,
                              leftIndent=15))  # Indent for addresses/transport details

    styles.add(ParagraphStyle(name='BodyText',
                              parent=styles['Normal'],
                              fontSize=10,
                              leading=12,
                              spaceAfter=6))

    story = []

    # 2. Add Title and Initial Info
    title_style = styles['Heading1']
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    # 3. Process Text Line-by-Line to Apply Styles
    for line in text.split("\n"):
        stripped_line = line.strip()

        if not stripped_line:
            # Use small spacer for paragraph break
            story.append(Spacer(1, 0.1 * inch))
            continue

        # --- Pattern Matching Logic ---
        # 1. Day Header (e.g., 'Day 1: 2025-06-01' or just '2025-06-01')
        if stripped_line.startswith('Day') or re.match(r'^\d{4}-\d{2}-\d{2}', stripped_line):
            story.append(Paragraph(stripped_line, styles['DayHeader']))
            continue

        # 2. Activity Header (Starts with time, e.g., '10:00 - Gardens by the Bay')
        # Use regex to find time format HH:MM at the start
        if re.match(r'^\d{1,2}:\d{2}', stripped_line):
            story.append(Paragraph(stripped_line, styles['ActivityHeader']))
            continue

        # 3. Transport/Address/Cost Details (e.g., 'Address:', 'Ride Duration:', 'Cost estimate:')
        if any(stripped_line.startswith(p) for p in
               ['Address:', 'Ride Duration:', 'Public Transport Duration:', 'Cycle Duration:', 'Cost estimate:',
                'Tags:']):
            # Bolding keywords within the DetailText to emphasize them
            line_formatted = stripped_line.replace('Address:', '<b>Address:</b>')
            line_formatted = line_formatted.replace('Ride Duration:', '<b>Ride Duration:</b>')
            line_formatted = line_formatted.replace('Public Transport Duration:', '<b>Public Transport Duration:</b>')
            line_formatted = line_formatted.replace('Cycle Duration:', '<b>Cycle Duration:</b>')
            story.append(Paragraph(line_formatted, styles['DetailText']))
            continue

        # 4. Fallback to Body Text (Descriptions, Route Summaries, etc.)
        else:
            story.append(Paragraph(stripped_line, styles['BodyText']))

    # 4. Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
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

def create_pdf_bytes_flowable(text, title="Itinerary Export"):
    # 1. Setup the document template and buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()
    story = []

    # 2. Add Title (using a large bold style)
    title_style = styles['Heading1']
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.25 * inch))

    # 3. Add Content (split by true paragraph breaks - '\n\n')
    body_style = styles['Normal']
    body_style.fontSize = 10
    body_style.leading = 12  # Line spacing (equivalent to your 'leading')

    for paragraph in text.split("\n\n"):
        if paragraph.strip():
            # Paragraph automatically handles word wrapping based on doc width
            p = Paragraph(paragraph.strip(), body_style)
            story.append(p)
            # Add small vertical space between paragraphs
            story.append(Spacer(1, 0.1 * inch))

            # 4. Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.read()