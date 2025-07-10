import streamlit as st
import os
import uuid
import whisper
import subprocess
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from transformers import pipeline
import yt_dlp

st.set_page_config(page_title="🎬 AI TikTok Clip Generator", layout="centered")

st.title("🎬 YouTube to TikTok Clip Generator")
st.markdown("Paste a YouTube video URL. The app will automatically:")
st.markdown("""
- Download the video  
- Transcribe with Whisper  
- Detect exciting highlights  
- Crop to 9:16 format  
- Add subtitles  
- Export TikTok-ready clips 🎉
""")

@st.cache_resource
def load_model():
    return whisper.load_model("base")

@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis")

def download_video(url, output_path):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def transcribe_video(video_path, model):
    result = model.transcribe(video_path)
    return result['segments']

def detect_highlights(segments, classifier):
    highlights = []
    for seg in segments:
        result = classifier(seg['text'])[0]
        if result['label'] == "POSITIVE" and result['score'] > 0.9:
            highlights.append(seg)
    return highlights

def crop_to_9x16(input_file, output_file, start, end):
    clip = VideoFileClip(input_file).subclip(start, end)
    h = clip.h
    w = int(h * 9 / 16)
    x_center = clip.w // 2
    x1 = max(0, x_center - w // 2)
    x2 = min(clip.w, x_center + w // 2)
    cropped = clip.crop(x1=x1, x2=x2)
    cropped = cropped.resize(height=1080)
    cropped.write_videofile(output_file, codec="libx264", audio_codec="aac", threads=4, logger=None)

def add_subtitles(video_path, text, output_path):
    clip = VideoFileClip(video_path)
    subtitle = TextClip(text, fontsize=48, color='white', bg_color='black', font='Arial-Bold')
    subtitle = subtitle.set_position(('center', 'bottom')).set_duration(clip.duration)
    final = CompositeVideoClip([clip, subtitle])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4, logger=None)

def generate_clips(youtube_url):
    video_id = str(uuid.uuid4())
    input_video = f"{video_id}.mp4"
    download_video(youtube_url, input_video)

    model = load_model()
    classifier = load_classifier()

    segments = transcribe_video(input_video, model)
    highlights = detect_highlights(segments, classifier)

    os.makedirs("tiktok_clips", exist_ok=True)
    output_clips = []

    for i, seg in enumerate(highlights[:3]):
        start = seg['start']
        end = seg['end']
        text = seg['text']
        base = f"tiktok_clips/{uuid.uuid4()}"
        cropped_path = f"{base}_cropped.mp4"
        final_path = f"{base}_final.mp4"

        crop_to_9x16(input_video, cropped_path, start, end)
        add_subtitles(cropped_path, text, final_path)
        output_clips.append(final_path)

    return output_clips

youtube_url = st.text_input("Paste YouTube video URL")

if st.button("Generate TikTok Clips"):
    if not youtube_url.strip():
        st.warning("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Processing..."):
            try:
                final_clips = generate_clips(youtube_url)
                st.success("Clips generated successfully!")
                for path in final_clips:
                    st.video(path)
                    with open(path, "rb") as f:
                        st.download_button("Download", f, file_name=os.path.basename(path))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
