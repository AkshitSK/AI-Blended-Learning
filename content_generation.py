from transformers import pipeline
import nltk
nltk.download('punkt')
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import heapq
import numpy as np
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from gtts import gTTS
from tempfile import NamedTemporaryFile
import textwrap
from fpdf import FPDF
from io import BytesIO
import os

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

class PDF(FPDF):
    def __init__(self):
        super().__init__()

def string_to_pdf(text):
    pdf = PDF()
    pdf.add_page()
    
    # Ensure the path to the font file is correct
    font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
    pdf.add_font('DejaVu', '', font_path, uni=True)
    pdf.set_font('DejaVu', '', 12)
    
    pdf.multi_cell(0, 10, text)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)  # Go to the beginning of the BytesIO buffer
    return pdf_output

def split_text_into_chunks(text, max_tokens=500):
    words = nltk.word_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for word in words:
        current_chunk.append(word)
        current_chunk_length += 1
        if current_chunk_length >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_chunk_length = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def chunk_text(text, tokenizer, max_length=1024):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def summarize_transcript(transcription):
    print(("Starting: Analyze transcript", "blue"))
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    transcription_chunks = chunk_text(transcription, tokenizer, max_length=512)
    full_summary = ""
    for i, chunk in enumerate(transcription_chunks):
        print((f"Summarizing chunk {i+1}/{len(transcription_chunks)}", "blue"))
        if len(chunk.split()) > 10:
            try:
                summary_result = summarizer(chunk, do_sample=False, min_length=50, max_length=150)
                full_summary += " " + summary_result[0]['summary_text']
            except:
                print((f"Error summarizing chunk {i+1}. Skipping.", "yellow"))

    print(("Generated full summary. Now extracting key points...", "green"))

    def preprocess(text):
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [sent.strip() for sent in sentences if len(sent.split()) > 3]

    def score_sentences(sentences):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.sum(tfidf_matrix.toarray(), axis=1)
        return scores

    sentences = preprocess(full_summary)
    scores = score_sentences(sentences)

    num_key_points = min(15, len(sentences))
    top_indices = heapq.nlargest(num_key_points, range(len(scores)), key=lambda i: scores[i])
    key_points = [f"{i+1}. {sentences[idx]}" for i, idx in enumerate(sorted(top_indices))]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Comprehensive Analysis of Transcription", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Full Summary", styles['Heading2']))
    wrapped_summary = textwrap.fill(full_summary, width=100)
    story.append(Paragraph(wrapped_summary, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Key Points", styles['Heading2']))
    for key_point in key_points:
        wrapped_key_point = textwrap.fill(key_point, width=100)
        story.append(Paragraph(wrapped_key_point, styles['Normal']))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

def translate_text_line_by_line(text, target_language_code):
    tokenizer.src_lang = "en_XX"
    lines = text.split('\n')
    translated_text = ""
    for line in lines:
        if line.strip():
            inputs = tokenizer(line, return_tensors="pt", max_length=512, truncation=True)
            translated_tokens = model.generate(inputs.input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code])
            translated_line = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_text += translated_line + "\n"
        else:
            translated_text += "\n"
    return translated_text.strip()

def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language, slow=False)
    tts_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    return tts_file.name
