### Dependencies

Ensure you have the required dependencies installed. You can install them using pip:
```bash
pip install streamlit numpy pydub transformers PyPDF2 gtts whisper moviepy av matplotlib streamlit-webrtc
```

### Import Libraries
```python
import streamlit as st
import numpy as np
import pydub
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from PyPDF2 import PdfReader
from gtts import gTTS
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from moviepy.editor import VideoFileClip
import time
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import matplotlib.pyplot as plt
```
These libraries provide functionalities such as model loading, PDF reading, text-to-speech conversion, audio processing, video processing, real-time audio processing, and plotting.

### Load the MBart Model and Tokenizer
```python
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```
Load the MBart model and tokenizer for translation tasks.

### Define Audio Processor Class for Real-time Audio Processing
```python
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.waveform = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.waveform = audio.mean(axis=1) if audio.ndim == 2 else audio
        return frame
```
This class handles real-time audio processing for displaying live waveforms.

### Plot Live Waveform
```python
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
```
This function plots the live waveform of the audio being processed.

### Function to Extract Text from PDF
```python
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"  # Ensure lines are separated
    return text.strip()
```
Extracts text from a given PDF file.

### Function to Translate Text Line by Line
```python
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
```
Translates text line by line using the MBart model.
Languages supported are:
Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

### Function to Convert Text to Speech
```python
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language, slow=False)
    tts_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    return tts_file.name
```
Converts the given text to speech using Google Text-to-Speech (gTTS).
Languages supported are:
{'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French', 'gu': 'Gujarati', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese', 'jw': 'Javanese', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'la': 'Latin', 'lv': 'Latvian', 'mk': 'Macedonian', 'ms': 'Malay', 'ml': 'Malayalam', 'mr': 'Marathi', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'si': 'Sinhala', 'sk': 'Slovak', 'sq': 'Albanian', 'sr': 'Serbian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'zh-CN': 'Chinese', 'zh-TW': 'Chinese (Mandarin/Taiwan)', 'zh': 'Chinese (Mandarin)'}


### Function to Remove Silence from Audio
```python
def remove_silence(input_file, output_file, silence_thresh=-40, min_silence_len=1000):
    audio = AudioSegment.from_mp3(input_file)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    output_audio = AudioSegment.empty()
    for chunk in chunks:
        output_audio += chunk
    output_audio.export(output_file, format="mp3")
```
Removes silence from an audio file.

### Function to Extract Audio from Video
```python
def extract_audio_from_video(video_file_path, output_audio_file_path):
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    temp_audio_path = "temp_audio.wav"
    audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
    audio = AudioSegment.from_wav(temp_audio_path)
    audio.export(output_audio_file_path, format="mp3")
    os.remove(temp_audio_path)
    print(f"Audio has been successfully extracted to {output_audio_file_path}")
```
Extracts audio from a video file.

### Function to Trim, Split, and Summarize Video Audio
```python
def trimSplitPredict(file_path):   
    # Extract audio from video
    vocals_path = "output_audio.mp3"
    extract_audio_from_video(file_path, vocals_path)

    # Remove silent areas from the extracted audio
    remove_silence(vocals_path, file_path)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization")

    # Load the Whisper model and transcribe
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
```
Trims, splits, transcribes, and summarizes the audio from a video file.

### Streamlit App Setup
```python
st.markdown("<h1 style='text-align: center; color: white;'>Generative AI for Blended Learning</h1>", unsafe_allow_html=True)

# Navbar options
options = ["AudioBook Generator", "Video Summarizer"]

# Sidebar for navigation
page = st.sidebar.radio("Navigate", options)
```
Sets up the Streamlit app with a title and a sidebar for navigation between different functionalities.

### AudioBook Generator Page
```python
if page == "AudioBook Generator":
    st.header("AudioBook Generator")
    
    st.write("Upload a PDF and convert it to speech in a selected language.")
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

            # Set up the webrtc streamer for live waveform display
            audio_processor = AudioProcessor()
            webrtc_ctx = webrtc_streamer(
                key="audio",
                mode=WebRtcMode.SENDRECV,
                audio_processor_factory=lambda: audio_processor,
                media_stream_constraints={"audio": True},
                async_processing=True,
            )

            # Plot and display the live waveform
            if webrtc_ctx.state.playing:
                plot_live_waveform(audio_processor)

            # Optionally, provide a download link for the audio file
            st.download_button(
                label="Download Translated Audio",
                data=audio_bytes,
                file_name="translated_audio.mp3",
                mime="audio/mp3"
            )

            # Optionally, provide a download link for the translated text
            st.download_button(
                label="Download Translated Text",
                data=translated_text,
                file_name="translated_text.txt",
                mime="text/plain"
            )
```
Handles the AudioBook Generator functionality, including PDF upload, text extraction, translation, text-to-speech conversion, and real-time audio waveform display.

### Video Summarizer Page
```python
elif page == "Video Summarizer":
    st.header("Video Summarizer")

    # Upload video file
    uploaded_video_file = st.file_uploader("Upload a Video file", type=["mp4", "mov", "avi"])
    if uploaded_video_file is not None

:
        # Save the uploaded video to a temporary file
        temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_file.write(uploaded_video_file.read())
        temp_video_file.close()

        st.write("Summarizing video...")
        # Call the video summarizer function
        summary = trimSplitPredict(temp_video_file.name)

        st.write("Summary completed!")
        st.text_area("Video Summary", summary, height=200)

        # Optionally, you can provide a download link for the video summary
        st.download_button(
            label="Download Video Summary",
            data=summary,
            file_name="video_summary.txt",
            mime="text/plain"
        )
```
Handles the Video Summarizer functionality, including video upload, audio extraction, silence removal, transcription, and summarization.

### Full Code Overview
This comprehensive Streamlit application combines various advanced functionalities such as real-time audio processing, text-to-speech, translation, and video summarization. It leverages multiple powerful libraries and models to provide an interactive user experience for generating audiobooks from PDFs and summarizing video content.