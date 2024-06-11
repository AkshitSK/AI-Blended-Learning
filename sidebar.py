import streamlit as st
import os
import queue

def initialize_session_state():
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""
    if 'video_ready' not in st.session_state:
        st.session_state.video_ready = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processed_time' not in st.session_state:
        st.session_state.processed_time = 0
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    if 'video_path' not in st.session_state:
        st.session_state.video_path = ""
    if 'progress_queue' not in st.session_state:
        st.session_state.progress_queue = queue.Queue()
    if 'transcript_queue' not in st.session_state:
        st.session_state.transcript_queue = queue.Queue()

def validate_chunks_folder():
    if not os.path.exists("./chunks"):
        os.makedirs("./chunks")
    else:
        for file in os.listdir("./chunks"):
            file_path = os.path.join("./chunks", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

def sidebar():
    initialize_session_state()
    validate_chunks_folder()

    page = st.sidebar.selectbox("Choose a page", ["Video Summarizer", "PDF Translator"])
    transcribe_button = False
    translate_button = False
    
    if page == "Video Summarizer":
        st.header("Video Summarizer")
        with st.sidebar:
            st.header("Video Processing")
            st.session_state.video_path = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
            transcribe_button = st.button("Process Video")

    if page == "PDF Translator":
        st.header("PDF Translator and Text-to-Speech")
        with st.sidebar:
            st.write("Upload a PDF and convert text to speech in a selected language.")
            st.session_state.pdf_path = st.file_uploader("Upload a PDF file", type=["pdf"])
            st.session_state.target_language = st.selectbox("Select the target language", ["hi_IN", "bn_IN", "ta_IN", "ml_IN"])
            translate_button = st.button("Translate")

    return transcribe_button, translate_button, st.sidebar.progress(0)