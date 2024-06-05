import tempfile
import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from PyPDF2 import PdfReader
from gtts import gTTS
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from moviepy.editor import VideoFileClip
import os
from fpdf import FPDF
import torch
import logging

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("No GPU available. Using CPU.")
    return device

device = setup_device()

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

@st.cache_data
def string_to_pdf(input_string):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    lines = input_string.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_file_path = tmp_file.name
    pdf.output(pdf_file_path)
    return pdf_file_path

@st.cache_data
def txt_to_pdf(txt_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    with open(txt_file, 'r') as file:
        for line in file:
            pdf.multi_cell(0, 10, line)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_file_path = tmp_file.name
    pdf.output(pdf_file_path)
    return pdf_file_path

@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

@st.cache_data
def translate_text_line_by_line(text, target_language_code):
    tokenizer.src_lang = "en_XX"
    lines = text.split('\n')
    translated_text = ""
    for line in lines:
        if line.strip():
            inputs = tokenizer(line, return_tensors="pt", max_length=512, truncation=True).to(device)
            translated_tokens = model.generate(inputs.input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code])
            translated_line = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_text += translated_line + "\n"
        else:
            translated_text += "\n"
    return translated_text.strip()

@st.cache_data
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language, slow=False)
    tts_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    return tts_file.name

@st.cache_data
def remove_silence(input_file, silence_thresh=-40, min_silence_len=1000):
    audio = AudioSegment.from_mp3(input_file)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    output_audio = AudioSegment.empty()
    for chunk in chunks:
        output_audio += chunk
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        output_audio_file = tmp_file.name
    output_audio.export(output_audio_file, format="mp3")
    return output_audio_file

@st.cache_data
def extract_audio_from_video(video_file_path):
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    temp_audio_path = "temp_audio.wav"
    audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
    audio = AudioSegment.from_wav(temp_audio_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        output_audio_file_path = tmp_file.name
    audio.export(output_audio_file_path, format="mp3")
    os.remove(temp_audio_path)
    return output_audio_file_path

@st.cache_data
def trim_split_predict(file_path):   
    vocals_path = extract_audio_from_video(file_path)
    summarizer = pipeline("summarization", device=0 if torch.cuda.is_available() else -1)
    model = whisper.load_model("medium").to(device)
    result = model.transcribe(vocals_path)
    transcription = result["text"]
    summary_result = summarizer(transcription, do_sample=False, max_length=50)
    summary_text = " ".join([summary['summary_text'] for summary in summary_result])
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        txt_file_path = tmp_file.name
    with open(txt_file_path, "w", encoding="utf-8") as txt:
        txt.write(summary_text)
    return summary_text, txt_file_path

def transcribe_and_display(file_path):
    st.write("Extracting audio from video...")
    vocals_path = extract_audio_from_video(file_path)
    st.write("Transcribing audio...")
    model = whisper.load_model("medium").to(device)
    transcription_placeholder = st.empty()
    chunk_size = 30  # seconds
    audio = AudioSegment.from_file(vocals_path)
    transcription = ""
    for i in range(0, len(audio), chunk_size * 1000):
        chunk = audio[i:i + chunk_size * 1000]
        temp_chunk_path = f"temp_chunk_{i // 1000}.wav"
        chunk.export(temp_chunk_path, format="wav")
        result = model.transcribe(temp_chunk_path)
        os.remove(temp_chunk_path)
        transcription += result["text"] + " "
        transcription_placeholder.text_area("Transcription", transcription, height=200)
        st.experimental_rerun()
    return transcription

def summarize_and_display(transcription):
    st.write("Summarizing transcription...")
    summarizer = pipeline("summarization", device=0 if torch.cuda.is_available() else -1)
    summary_points = []
    chunk_size = 512
    for i in range(0, len(transcription), chunk_size):
        chunk = transcription[i:i+chunk_size]
        summary_result = summarizer(chunk, do_sample=False, max_length=50)
        for summary in summary_result:
            summary_points.append(summary['summary_text'])
            st.write(f"Summary Chunk {i+1}: {summary['summary_text']}")
    summary_text = "\n".join(summary_points)
    return summary_text

st.markdown("<h1 style='text-align: center; color: white;'>Generative AI for Blended Learning</h1>", unsafe_allow_html=True)

page = st.sidebar.selectbox("Choose a page", ["Video Summarizer", "PDF Translator"])

if page == "PDF Translator":
    st.header("PDF Translator and Text-to-Speech")
    with st.sidebar:
        st.write("Upload a PDF and convert text to speech in a selected language.")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        target_language = st.selectbox("Select the target language", ["hi_IN", "bn_IN", "ta_IN", "ml_IN"])
        translate_button = st.button("Translate")

    if translate_button and uploaded_file is not None:
        st.write("Extracting text from PDF...")
        text = extract_text_from_pdf(uploaded_file)
        if not text:
            st.error("Could not extract text from the PDF file. Please try a different file.")
        else:
            st.write("Text extracted successfully!")
            st.write("Source Text:")
            st.text_area("Source Text", text, height=200)
            st.write("Translating text...")
            translated_text = translate_text_line_by_line(text, target_language)
            st.write("Translation completed!")
            st.text_area("Translated Text", translated_text, height=200)
            audio_file_path = text_to_speech(translated_text, target_language[:2])
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
            st.download_button(label="Download Translated Audio", data=audio_bytes, file_name="translated_audio.mp3", mime="audio/mp3")
            st.download_button(label="Download Translated Text", data=translated_text, file_name="translated_text.txt", mime="text/plain")
            pdf_file_path = string_to_pdf(translated_text)
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
            st.download_button(label='Download PDF', data=pdf_bytes, file_name='translated_text.pdf', mime='application/pdf')
            os.remove(pdf_file_path)
        st.cache_data.clear()
        st.cache_resource.clear()

elif page == "Video Summarizer":
    st.header("Video Summarizer")
    with st.sidebar:
        st.write("Upload a Video to summarize.")
        uploaded_video_file = st.file_uploader("Upload a Video file", type=["mp4", "mov", "avi"])
    
    if uploaded_video_file is not None:
        temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_file.write(uploaded_video_file.read())
        temp_video_file.close()
        transcription = transcribe_and_display(temp_video_file.name)
        st.write("Transcription completed!")
        st.text_area("Full Transcription", transcription, height=200)
        summary = summarize_and_display(transcription)
        st.write("Summary completed!")
        st.text_area("Video Summary", summary, height=200)
        pdf_file_path = string_to_pdf(summary)
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(label='Download PDF', data=pdf_bytes, file_name='summary.pdf', mime='application/pdf')
        os.remove(pdf_file_path)
        st.cache_data.clear()
        st.cache_resource.clear()