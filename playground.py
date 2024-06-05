import os
import re
import heapq
import numpy as np
import tempfile
import shutil
import atexit
import glob
from tqdm import tqdm
from termcolor import colored
import torch
from transformers import (
    MBart50TokenizerFast,
    pipeline,
    TRANSFORMERS_CACHE
)
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub import AudioSegment
import whisper
from moviepy.editor import VideoFileClip
from fpdf import FPDF

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
        print(colored(f"Using GPU: {torch.cuda.get_device_name(0)}", "green"))
    else:
        device = torch.device("cpu")
        print(colored("No GPU available. Using CPU.", "yellow"))
    return device

device = setup_device()


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
        print(colored("Downloaded models:", "blue"))
        for path in model_paths:
            print(f"  - {path}")
    else:
        print(colored("No downloaded models found.", "green"))

def delete_model(model_name):
    cache_dir = TRANSFORMERS_CACHE
    model_dir = os.path.join(cache_dir, model_name.replace("/", "--"))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(colored(f"Deleted model: {model_name}", "green"))
    else:
        print(colored(f"Model not found: {model_name}", "yellow"))

def delete_all_models():
    model_paths = get_model_paths()
    if model_paths:
        for path in model_paths:
            try:
                os.remove(path)
                print(colored(f"Deleted: {path}", "green"))
            except:
                print(colored(f"Failed to delete: {path}", "red"))
        
        # Also remove empty directories
        cache_dir = TRANSFORMERS_CACHE
        for root, dirs, _ in os.walk(cache_dir, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.rmdir(dir_path)
                except:
                    pass
        print(colored("All models deleted.", "green"))
    else:
        print(colored("No models to delete.", "yellow"))

def delete_whisper_model():
    home_dir = os.path.expanduser("~")
    whisper_dir = os.path.join(home_dir, ".cache", "whisper")
    if os.path.exists(whisper_dir):
        shutil.rmtree(whisper_dir)
        print(colored("Deleted Whisper model.", "green"))
    else:
        print(colored("Whisper model not found.", "yellow"))

def find_video_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                return os.path.join(root, file)
    return None

def extract_audio_from_video(video_file_path):
    print(colored("Starting: Extract audio from video", "blue"))
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
    print(colored("Completed: Extract audio from video", "green"))
    return output_audio_file_path

def transcribe_audio(audio_file_path):
    print(colored("Starting: Transcribe audio", "blue"))
    model = whisper.load_model("base").to(device)
    result = model.transcribe(audio_file_path)
    print(colored("Completed: Transcribe audio", "green"))
    print("\nTranscription:")
    print(result["text"])
    return result["text"]

def chunk_text(text, tokenizer, max_length=1024):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def analyze_transcript(transcription):
    print(colored("Starting: Analyze transcript", "blue"))
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    # Try to use GPU, but switch to CPU if there's an issue
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        print(colored("Using GPU for summarization", "green"))
    except:
        print(colored("CUDA error encountered. Switching to CPU.", "yellow"))
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    # Generate a comprehensive summary
    transcription_chunks = chunk_text(transcription, tokenizer, max_length=1024)
    full_summary = ""
    for i, chunk in enumerate(transcription_chunks):
        print(colored(f"Summarizing chunk {i+1}/{len(transcription_chunks)}", "blue"))
        summary_result = summarizer(chunk, do_sample=False, min_length=100)
        full_summary += " " + summary_result[0]['summary_text']

    print(colored("Generated full summary. Now extracting key points...", "green"))

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

    print(colored("Completed: Analyze transcript", "green"))

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
    print(colored(f"Comprehensive analysis has been saved to: {pdf_output_path}", "green"))

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_file = find_video_file(current_dir)
    
    if video_file:
        print(colored(f"Found video file: {video_file}", "green"))
        audio_file = extract_audio_from_video(video_file)
        transcription = transcribe_audio(audio_file)
        analyze_transcript(transcription)

        # After analysis, print and delete models
        print("\n" + "="*50)
        print_downloaded_models()
        print("="*50 + "\n")

        print("Deleting models...")
        delete_model("facebook/mbart-large-50-many-to-many-mmt")
        delete_model("facebook/bart-large-cnn")
        delete_whisper_model()
        print("Model cleanup completed.")
    else:
        print(colored("No video file found in the directory.", "red"))

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()