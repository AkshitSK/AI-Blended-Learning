import streamlit as st
import threading
import queue
from video_processing import format_time, split_video, process_chunks_async
from sidebar import sidebar
from content_generation import summarize_transcript, translate_text_line_by_line, text_to_speech, string_to_pdf
from PyPDF2 import PdfReader
import os


def update_transcript_text():
    transcript_text = ""
    current_transcriptions = [
        (start_time, end_time, transcription)
        for start_time, end_time, transcription in st.session_state.transcriptions
    ]
    for start_time, end_time, transcription in current_transcriptions:
        transcript_text += f"{format_time(start_time)} - {format_time(end_time)}: {transcription}\n"

    st.session_state.transcript_text = transcript_text

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def pdf_to_text(pdf_buffer):
    pdf_reader = PdfReader(pdf_buffer)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def main():
    st.title("Video Upload and Transcription")

    transcribe_button, translate_button, progress_bar = sidebar()
    video_placeholder = st.empty()
    transcript_placeholder = st.empty()
    summary_placeholder = st.empty()

    if transcribe_button:
        st.session_state.transcriptions = []
        st.session_state.transcript_text = ""
        st.session_state.last_transcription_time = -1
        st.session_state.video_ready = False
        st.session_state.processing_complete = False
        st.session_state.processed_time = 0
        st.session_state.current_time = 0

        # Save the uploaded file to a temporary location
        with open("temp_video.mp4", "wb") as f:
            f.write(st.session_state.video_path.getbuffer())

        chunks = split_video("temp_video.mp4")
        total_duration = sum(chunk[2] - chunk[1] for chunk in chunks)
        
        progress_queue = queue.Queue()
        transcript_queue = queue.Queue()

        processing_thread = threading.Thread(target=process_chunks_async, args=(chunks, total_duration, progress_queue, transcript_queue))
        processing_thread.start()

        status_placeholder = st.sidebar.empty()

        while not st.session_state.processing_complete:
            try:
                progress_update = progress_queue.get(timeout=1)
                if progress_update is None:
                    st.session_state.processing_complete = True
                    break
                processed_time, total_duration = progress_update
                progress = processed_time / total_duration
                progress_bar.progress(progress)
                status_placeholder.text(f"Processing Video {format_time(processed_time)}/{format_time(total_duration)}")

                st.session_state.video_ready = True

                transcriptions = transcript_queue.get(timeout=1)
                st.session_state.transcriptions = sorted(transcriptions, key=lambda x: x[0])

            except queue.Empty:
                continue

        st.session_state.processing_complete = True
        print("Transcribing complete")

    if st.session_state.video_ready:
        with video_placeholder.container():
            st.video("temp_video.mp4", format="video/mp4")

        st.session_state.current_time = 0

        if "transcript_text" not in st.session_state:
            st.session_state.transcript_text = ""

        update_transcript_text()
        transcript_placeholder.text_area("Transcript", st.session_state.transcript_text, height=400, key=f"transcript_text_area_{st.session_state.current_time}", disabled=True)

        if st.button("Summarize Transcript"):
            summary_buffer = summarize_transcript(st.session_state.transcript_text)
            summary = pdf_to_text(summary_buffer)

            summary_placeholder.text_area("Summary", summary, height=400, disabled=True)

            # Create a download button for the generated PDF
            st.download_button(
                label="Download PDF",
                data=summary_buffer.getvalue(),
                file_name="generated_pdf.pdf",
                mime="application/pdf"
            )
    
    if translate_button:
        st.write("Extracting text from PDF...")
        text = extract_text_from_pdf(st.session_state.pdf_path)
        if not text:
            st.error("Could not extract text from the PDF file. Please try a different file.")
        else:
            st.write("Text extracted successfully!")
            st.write("Source Text:")
            st.text_area("Source Text", text, height=200)
            st.write("Translating text...")
            translated_text = translate_text_line_by_line(text, st.session_state.target_language)
            st.write("Translation completed!")
            st.text_area("Translated Text", translated_text, height=200)
            audio_file_path = text_to_speech(translated_text, st.session_state.target_language[:2])
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

if __name__ == "__main__":
    main()
