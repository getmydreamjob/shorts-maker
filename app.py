import streamlit as st
import os
import uuid
import whisper
import subprocess
from transformers import pipeline
import yt_dlp

st.set_page_config(page_title="🎬 TikTok Shorts Generator", layout="centered")
st.title("🎬 AI TikTok Shorts Maker")

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis")

def download_youtube_video(url, output_path):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def transcribe_audio(video_path, model):
    return model.transcribe(video_path)['segments']

def detect_highlights(segments, sentiment_model):
    highlights = []
    for s in segments:
        result = sentiment_model(s['text'])[0]
        if result['label'] == 'POSITIVE' and result['score'] > 0.9:
            highlights.append(s)
    return highlights

def crop_and_generate(video_path, segment, index):
    start = segment['start']
    end = segment['end']
    text = segment['text'].replace('"', '').replace("'", "")

    uid = uuid.uuid4().hex
    output_dir = "tiktok_clips"
    os.makedirs(output_dir, exist_ok=True)
    subtitle_file = f"{output_dir}/subs_{uid}.srt"
    final_output = f"{output_dir}/final_{uid}.mp4"

    # Generate SRT subtitle file
    with open(subtitle_file, "w") as f:
        f.write(f"1\n00:00:00,000 --> 00:00:{int(end - start)*1000}\n{text}\n")

    # Crop to 9:16 with face area centered (horizontal crop) and overlay subtitles
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-vf", "crop='ih*9/16:ih:(iw-ih*9/16)/2:0',subtitles=" + subtitle_file,
        "-preset", "ultrafast",
        "-c:a", "aac",
        final_output
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return final_output

def generate_tiktok_clips(url):
    video_file = f"{uuid.uuid4().hex}.mp4"
    download_youtube_video(url, video_file)

    whisper_model = load_whisper()
    classifier = load_classifier()

    segments = transcribe_audio(video_file, whisper_model)
    highlights = detect_highlights(segments, classifier)

    output_paths = []
    for i, h in enumerate(highlights[:3]):  # Limit to top 3 highlights
        path = crop_and_generate(video_file, h, i)
        output_paths.append(path)

    return output_paths

# UI
video_url = st.text_input("Paste YouTube Video URL")
if st.button("Generate TikTok Clips"):
    if not video_url:
        st.warning("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Processing..."):
            try:
                paths = generate_tiktok_clips(video_url)
                st.success("Done! Download your TikTok clips below.")
                for path in paths:
                    st.video(path)
                    with open(path, "rb") as f:
                        st.download_button("Download", f, file_name=os.path.basename(path))
            except Exception as e:
                st.error(f"Error: {e}")
