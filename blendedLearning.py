import tempfile
import streamlit as st
import numpy as np
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from PyPDF2 import PdfReader
from gtts import gTTS
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from moviepy.editor import VideoFileClip
import os
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import matplotlib.pyplot as plt
from fpdf import FPDF


# Load the MBart model and tokenizer for translation
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Audio Processor class for real-time audio processing
# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.waveform = None

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         audio = frame.to_ndarray()
#         self.waveform = audio.mean(axis=1) if audio.ndim == 2 else audio
#         return frame

# Function to plot live waveform
def plot_live_waveform(audio_processor):
    if audio_processor.waveform is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(audio_processor.waveform)
        plt.title("Live Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.tight_layout()
        st.pyplot(plt)

def string_to_pdf(input_string, pdf_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Split the input string into lines
    lines = input_string.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    
    pdf.output(pdf_file)

def txt_to_pdf(txt_file, pdf_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    with open(txt_file, 'r') as file:
        for line in file:
            pdf.multi_cell(0, 10, line)

    pdf.output(pdf_file)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"  # Ensure lines are separated
    return text.strip()

# Function to translate text line by line using the MBart model
def translate_text_line_by_line(text, target_language_code):
    tokenizer.src_lang = "en_XX"
    lines = text.split('\n')
    translated_text = ""
    for line in lines:
        if line.strip():  # Skip empty lines
            inputs = tokenizer(line, return_tensors="pt", max_length=512, truncation=True)
            translated_tokens = model.generate(inputs.input_ids, forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code])
            translated_line = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_text += translated_line + "\n"
        else:
            translated_text += "\n"
    return translated_text.strip()

# Function to convert text to speech using gTTS
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language, slow=False)
    tts_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    return tts_file.name

# Function to remove silence from an audio file
def remove_silence(input_file, output_file, silence_thresh=-40, min_silence_len=1000):
    audio = AudioSegment.from_mp3(input_file)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    output_audio = AudioSegment.empty()
    for chunk in chunks:
        output_audio += chunk
    output_audio.export(output_file, format="mp3")

# Function to extract audio from a video file
def extract_audio_from_video(video_file_path, output_audio_file_path):
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    temp_audio_path = "temp_audio.wav"
    audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
    audio = AudioSegment.from_wav(temp_audio_path)
    audio.export(output_audio_file_path, format="mp3")
    os.remove(temp_audio_path)
    print(f"Audio has been successfully extracted to {output_audio_file_path}")

# Function to trim, split, and predict the summary for a video file
def trimSplitPredict(file_path):   
    # Extract audio from video
    vocals_path = "output_audio.mp3"
    extract_audio_from_video(file_path, vocals_path)

    # Remove silent areas from the extracted audio
    remove_silence(vocals_path, file_path)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization")

    # Load the Whisper model and transcribe the audio
    model = whisper.load_model("medium")
    result = model.transcribe(vocals_path)
    transcription = result["text"]

    # Summarize the transcription
    summary_result = summarizer(transcription, do_sample=False)
    
    # Extract text from the summarizer output
    summary_text = " ".join([summary['summary_text'] for summary in summary_result])

    # Write the summary to a text file
    with open("summary.txt", "w", encoding="utf-8") as txt:
        txt.write(summary_text)
    
    return summary_text

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: white;'>Generative AI for Blended Learning</h1>", unsafe_allow_html=True)

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["PDF Translator", "Video Summarizer"])

if page == "PDF Translator":
    st.header("PDF Translator and Text-to-Speech")

    # Sidebar with expanders for PDF Translator
    with st.sidebar:
        st.write("Upload a PDF and convert text to speech in a selected language.")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        target_language = st.selectbox(
            "Select the target language",
            ["hi_IN", "bn_IN", "ta_IN", "ml_IN"]  # MBart language codes
        )
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

            # Convert translated text to speech
            audio_file_path = text_to_speech(translated_text, target_language[:2])

            # Provide audio playback
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

            # # Set up the webrtc streamer for live waveform display
            # audio_processor = AudioProcessor()
            # webrtc_ctx = webrtc_streamer(
            #     key="audio",
            #     mode=WebRtcMode.SENDRECV,
            #     audio_processor_factory=lambda: audio_processor,
            #     media_stream_constraints={"audio": True},
            #     async_processing=True,
            # )

            # # Plot and display the live waveform
            # if webrtc_ctx.state.playing:
            #     plot_live_waveform(audio_processor)

            # Provide a download link for the audio file
            st.download_button(
                label="Download Translated Audio",
                data=audio_bytes,
                file_name="translated_audio.mp3",
                mime="audio/mp3"
            )

            # Provide a download link for the translated text
            st.download_button(
                label="Download Translated Text",
                data=translated_text,
                file_name="translated_text.txt",
                mime="text/plain"
            )

            # if translated_text:
            #     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            #         pdf_file_path = tmp_file.name
                
            #     string_to_pdf(translated_text, pdf_file_path)
                
            #     with open(pdf_file_path, 'rb') as pdf_file:
            #         pdf_bytes = pdf_file.read()
                
            #     st.download_button(
            #         label='Download PDF',
            #         data=pdf_bytes,
            #         file_name='translated_text.pdf',
            #         mime='application/pdf'
            #     )
                
            #     # Clean up temporary file
            #     os.remove(pdf_file_path)

# Video Summarizer functionality
elif page == "Video Summarizer":
    st.header("Video Summarizer")
    with st.sidebar:
        st.write("Upload a Video to summarize.")
        uploaded_video_file = st.file_uploader("Upload a Video file", type=["mp4", "mov", "avi"])
    
    if uploaded_video_file is not None:
        # Save the uploaded video to a temporary file
        temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_file.write(uploaded_video_file.read())
        temp_video_file.close()

        st.write("Summarizing video...")
        # Call the video summarizer function
        summary = trimSplitPredict(temp_video_file.name)

        st.write("Summary completed!")
        st.text_area("Video Summary", summary, height=200)

        # Provide a download link for the video summary
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_file_path = tmp_file.name
        
        string_to_pdf(summary, pdf_file_path)
        
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        st.download_button(
            label='Download PDF',
            data=pdf_bytes,
            file_name='summary.pdf',
            mime='application/pdf'
        )
        
        # Clean up temporary file
        os.remove(pdf_file_path)
