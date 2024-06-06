import os
import re
import heapq
import numpy as np
import tempfile
import shutil
import atexit
import glob
import torch
from transformers import MBart50TokenizerFast, pipeline, TRANSFORMERS_CACHE
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub import AudioSegment
import whisper
from moviepy.editor import VideoFileClip
from fpdf import FPDF
import concurrent.futures
import streamlit as st

# Force the use of GTX 1650 CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Assuming GTX 1650 is the first GPU

# List to keep track of temporary files
temp_files = []

def cleanup():
    for file_path in temp_files:
        try:
            os.remove(file_path)
        except:
            pass
    for file_path in glob.glob("temp_chunk_*.wav"):
        try:
            os.remove(file_path)
        except:
            pass

atexit.register(cleanup)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, f"Using GPU: {torch.cuda.get_device_name(0)}"
    else:
        device = torch.device("cpu")
        return device, "No GPU available. Using CPU."

device, device_message = setup_device()

def get_model_paths():
    cache_dir = TRANSFORMERS_CACHE
    model_paths = []
    for root, _, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.bin'):  # Model weights are typically .bin files
                model_paths.append(os.path.join(root, file))
    return model_paths

def print_downloaded_models():
    model_paths = get_model_paths()
    if model_paths:
        return ["Downloaded models:"] + model_paths
    else:
        return ["No downloaded models found."]

def delete_model(model_name):
    cache_dir = TRANSFORMERS_CACHE
    model_dir = os.path.join(cache_dir, model_name.replace("/", "--"))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        return f"Deleted model: {model_name}"
    else:
        return f"Model not found: {model_name}"

def delete_all_models():
    model_paths = get_model_paths()
    messages = []
    if model_paths:
        for path in model_paths:
            try:
                os.remove(path)
                messages.append(f"Deleted: {path}")
            except:
                messages.append(f"Failed to delete: {path}")
        
        # Also remove empty directories
        cache_dir = TRANSFORMERS_CACHE
        for root, dirs, _ in os.walk(cache_dir, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.rmdir(dir_path)
                except:
                    pass
        messages.append("All models deleted.")
    else:
        messages.append("No models to delete.")
    return messages

def delete_whisper_model():
    home_dir = os.path.expanduser("~")
    whisper_dir = os.path.join(home_dir, ".cache", "whisper")
    if os.path.exists(whisper_dir):
        shutil.rmtree(whisper_dir)
        return "Deleted Whisper model."
    else:
        return "Whisper model not found."

def find_video_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                return os.path.join(root, file)
    return None

def extract_audio_from_video(video_file_path):
    st.write("Starting: Extract audio from video")
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    temp_audio_path = "temp_audio.wav"
    audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
    audio = AudioSegment.from_wav(temp_audio_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        output_audio_file_path = tmp_file.name
        temp_files.append(output_audio_file_path)
    audio.export(output_audio_file_path, format="mp3")
    os.remove(temp_audio_path)
    st.write("Completed: Extract audio from video")
    return output_audio_file_path

def transcribe_chunk(chunk_path, index):
    try:
        model = whisper.load_model("base").to(device)
        result = model.transcribe(chunk_path)
        return index, result["text"], None
    except Exception as e:
        return index, "", str(e)

def transcribe_audio(audio_file_path, chunk_size=30):  # 30 seconds chunks
    st.write("Starting: Transcribe audio")
    audio = AudioSegment.from_file(audio_file_path)
    
    chunks = []
    for i in range(0, len(audio), chunk_size * 1000):
        chunk = audio[i:i + chunk_size * 1000]
        temp_chunk_path = f"temp_chunk_{i // 1000}.wav"
        chunk.export(temp_chunk_path, format="wav")
        temp_files.append(temp_chunk_path)
        chunks.append((temp_chunk_path, i // (chunk_size * 1000)))

    transcription_parts = [None] * len(chunks)
    errors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(transcribe_chunk, *chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            index, text, error = future.result()
            if error:
                errors.append(f"Error transcribing chunk {index}: {error}")
            else:
                transcription_parts[index] = text
            
    for chunk_path, _ in chunks:
        try:
            os.remove(chunk_path)
        except:
            pass

    transcription = " ".join(filter(None, transcription_parts))
    return transcription, errors

def chunk_text(text, tokenizer, max_length=1024):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def analyze_transcript(transcription):
    st.write("Starting: Analyze transcript")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    # Try to use GPU, but switch to CPU if there's an issue
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        summary_device_message = "Using GPU for summarization"
    except:
        summary_device_message = "CUDA error encountered. Switching to CPU."
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    st.write(summary_device_message)

    # Generate a comprehensive summary
    transcription_chunks = chunk_text(transcription, tokenizer, max_length=512)  # Smaller chunks for summarization too
    full_summary = ""
    for i, chunk in enumerate(transcription_chunks):
        if len(chunk.split()) > 10:  # Only summarize if chunk is not too short
            try:
                summary_result = summarizer(chunk, do_sample=False, min_length=50, max_length=150)
                full_summary += " " + summary_result[0]['summary_text']
            except:
                pass

    # Function to preprocess text
    def preprocess(text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split into sentences
        return [sent.strip() for sent in sentences if len(sent.split()) > 3]  # Filter out very short sentences

    # Function to calculate sentence scores using TF-IDF
    def score_sentences(sentences):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.sum(tfidf_matrix.toarray(), axis=1)
        return scores

    # Preprocess and score sentences
    sentences = preprocess(full_summary)
    scores = score_sentences(sentences)

    # Select top sentences as key points
    num_key_points = min(15, len(sentences))  # Up to 15 key points
    top_indices = heapq.nlargest(num_key_points, range(len(scores)), key=lambda i: scores[i])
    key_points = [f"{i+1}. {sentences[idx]}" for i, idx in enumerate(sorted(top_indices))]

    st.write("Completed: Analyze transcript")

    # Save key points and full summary to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Comprehensive Analysis of Transcription", ln=True, align='C')

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, "Full Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, full_summary)

    pdf.add_page()
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, "Key Points", ln=True)
    pdf.set_font("Arial", size=12)
    for key_point in key_points:
        pdf.multi_cell(0, 10, key_point)

    pdf_output_path = "transcript_analysis.pdf"
    pdf.output(pdf_output_path)
    return full_summary, key_points, pdf_output_path

def main():
    st.title("Video Transcription and Analysis")

    with st.sidebar.expander("Options"):
        st.write("Configure your options here.")
        video_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"])
        delete_models = st.checkbox("Delete Models after Analysis", value=True)
    
    if st.button("Start Analysis"):
        if video_file is not None:
            with st.spinner("Processing..."):
                temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(video_file.getbuffer())
                
                st.write(f"Uploaded video file: {video_file.name}")
                audio_file = extract_audio_from_video(temp_video_path)

                transcription, transcription_errors = transcribe_audio(audio_file)
                st.write("Completed: Transcribe audio")
                st.write("Transcription:")
                st.write(transcription)
                if transcription_errors:
                    st.write("Errors during transcription:")
                    for error in transcription_errors:
                        st.write(error)

                full_summary, key_points, pdf_output_path = analyze_transcript(transcription)

                st.write("Full Summary:")
                st.write(full_summary)

                st.write("Key Points:")
                for key_point in key_points:
                    st.write(key_point)

                st.write(f"Comprehensive analysis has been saved to: {pdf_output_path}")

                if delete_models:
                    st.write("\n" + "="*50)
                    model_paths = print_downloaded_models()
                    for path in model_paths:
                        st.write(path)
                    st.write("="*50 + "\n")

                    st.write("Deleting models...")
                    st.write(delete_model("facebook/mbart-large-50-many-to-many-mmt"))
                    st.write(delete_model("facebook/bart-large-cnn"))
                    st.write(delete_whisper_model())
                    st.write("Model cleanup completed.")
        else:
            st.warning("Please upload a video file.")

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
