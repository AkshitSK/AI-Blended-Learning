import moviepy.editor as mp
import queue
import threading
from faster_whisper import WhisperModel

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def split_video(video_path):
    try:
        video = mp.VideoFileClip(video_path)
        duration = int(video.duration)
        chunks = []
        chunk_size = 831  # seconds
        start_time = 0
        while start_time < duration:
            end_time = min(start_time + chunk_size, duration)
            chunk_path = f"./chunks/chunk_{start_time}_{end_time}.wav"
            video.audio.subclip(start_time, end_time).write_audiofile(chunk_path)
            chunks.append((chunk_path, start_time, end_time))
            start_time = end_time
        return chunks
    except Exception as e:
        print(f"Error in split_video: {e}")
        raise

def process_chunk(chunk, result_queue):
    print("In process_chunk")
    try:
        model_size = "large-v3"
        
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        results, info = model.transcribe(chunk[0], beam_size=5)

        text = ''.join([segment.text for segment in results])

        result_queue.put((chunk[1], chunk[2], text))
    except Exception as e:
        print(f"Error in process_chunk: {e}")
        result_queue.put((chunk[1], chunk[2], f"Error: {e}"))

def process_chunks_async(chunks, total_duration, progress_queue, transcript_queue):
    try:
        result_queue = queue.Queue()
        threads = []
        for chunk in chunks:
            thread = threading.Thread(target=process_chunk, args=(chunk, result_queue))
            thread.start()
            threads.append(thread)

        processed_time = 0
        transcriptions = []
        while len(transcriptions) < len(chunks):
            start_time, end_time, text = result_queue.get()
            transcriptions.append((start_time, end_time, text))
            processed_time += end_time - start_time
            progress_queue.put((processed_time, total_duration))
            transcript_queue.put(transcriptions)

        for thread in threads:
            thread.join()

        progress_queue.put(None)
        transcript_queue.put(None)
    except Exception as e:
        print(f"Error in process_chunks_async: {e}")
        progress_queue.put(None)
        transcript_queue.put(None)