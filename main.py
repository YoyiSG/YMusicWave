#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import streamlit as st
from moviepy.editor import AudioFileClip, VideoClip, VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from packaging import version
import moviepy

# ============ 工具函数 ============
def hex_to_bgr(color: str):
    if "," in color:
        r, g, b = map(int, color.split(","))
        return b, g, r
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def blur_image(img: np.ndarray, ratio: float):
    """高斯虚化；ratio∈[0,0.2]，内部做二次方平滑"""
    if ratio <= 0:
        return img.copy()
    k = max(
        3,
        int(min(img.shape[:2]) * (ratio**1.5)) | 1,  # 二次方降低模糊力度
    )
    return cv2.GaussianBlur(img, (k, k), 0)


def prepare_spectrum(audio_path: str, n_bars: int, fps: int, sr_out=44100, n_fft=2048):
    y, sr = librosa.load(audio_path, sr=sr_out, mono=True)
    hop_length = int(sr / fps)
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    edges = np.linspace(freqs.min(), freqs.max(), n_bars + 1)
    bars = np.zeros((spec.shape[1], n_bars), np.float32)
    for i in range(n_bars):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        if idx.size:
            bars[:, i] = spec[idx].mean(axis=0)
    bars /= bars.max() + 1e-9
    return bars


# ============ 合成函数 ============
def generate_video(
    audio_path: str,
    image_path: str,
    output_path: str,
    *,
    n_bars=64,
    alpha=0.7,
    blur_ratio=0.05,
    fps=30,
    resolution=None,
    bar_color="#20c3ff",
    bar_height_ratio=0.9,
    bar_bottom_margin=0.0,
    bar_width_ratio=0.85,
    amp_scale=1.5,
    progress_callback=None,
):
    bg = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if resolution:
        bg = cv2.resize(bg, resolution, interpolation=cv2.INTER_AREA)
    h, w = bg.shape[:2]
    bg_blur = blur_image(bg, blur_ratio)

    bars = prepare_spectrum(audio_path, n_bars, fps)
    n_frames = bars.shape[0]

    bar_slot_w = w / n_bars
    bar_draw_w = int(bar_slot_w * bar_width_ratio)
    bar_max_h = int(h * bar_height_ratio)
    bottom_offset = int(h * bar_bottom_margin)
    color_bgr = hex_to_bgr(bar_color)
    alpha = np.clip(alpha, 0.0, 1.0)

    def make_frame(t):
        frame_idx = min(int(t * fps), n_frames - 1)
        bar_vals = bars[frame_idx]

        canvas = bg_blur.copy()
        for i, val in enumerate(bar_vals):
            bar_h = int(min(val * bar_max_h * amp_scale, bar_max_h))
            x1 = int(i * bar_slot_w + (bar_slot_w - bar_draw_w) / 2)
            x2 = x1 + bar_draw_w
            y1 = h - bar_h - bottom_offset
            y2 = h - bottom_offset
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    video_clip = VideoClip(make_frame, duration=duration).set_audio(audio_clip).set_fps(
        fps
    )

    common = dict(
        codec="libx264",
        audio_codec="aac",
        threads=os.cpu_count(),
        preset="medium",
    )

    # MoviePy ≥ 2：直接带回调
    if version.parse(moviepy.__version__).major >= 2:
        def _cb(c, t):
            if progress_callback:
                progress_callback(c / t)

        video_clip.write_videofile(
            output_path,
            **common,
            progress_bar=False,
            logger=None,
            callback=_cb,
        )
        if progress_callback:
            progress_callback(1.0)
        return

    # MoviePy 1.x：手写进度
    temp_dir = Path(tempfile.mkdtemp())
    tmp_noaudio = temp_dir / "tmp_noaudio.mp4"

    writer = FFMPEG_VideoWriter(
        str(tmp_noaudio),
        size=video_clip.size,
        fps=fps,
        codec="libx264",
        preset="medium",
    )
    total_frames = int(duration * fps)

    for i, frame in enumerate(video_clip.iter_frames(fps=fps, dtype="uint8")):
        writer.write_frame(frame)
        if progress_callback and i % max(1, total_frames // 100) == 0:
            progress_callback(i / total_frames)
    writer.close()

    (VideoFileClip(str(tmp_noaudio))
        .set_audio(audio_clip)
        .write_videofile(output_path, **common))

    if progress_callback:
        progress_callback(1.0)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============ Streamlit UI ============
st.set_page_config(page_title="Audio Visualizer", layout="wide")
st.title("🎧 音频可视化视频生成器")

with st.sidebar:
    st.header("⚙️ 参数设置")
    n_bars = st.slider("柱状条数量", 16, 256, 64, step=8)
    alpha = st.slider("柱状条透明度", 0.0, 1.0, 0.7, step=0.05)
    blur_ratio = st.slider("背景虚化比例", 0.0, 0.2, 0.05, step=0.01)
    fps = st.slider("帧率 (FPS)", 15, 60, 30, step=1)
    bar_color = st.color_picker("柱状条颜色", "#20c3ff")

    st.subheader("📏 尺寸 / 位置")
    bar_height_ratio = st.slider("高度比例", 0.1, 1.0, 0.9, step=0.05)
    bar_bottom_margin = st.slider("底部边距比例", 0.0, 0.5, 0.0, step=0.02)
    bar_width_ratio = st.slider("宽度比例(槽内)", 0.1, 1.0, 0.85, step=0.05)
    amp_scale = st.slider("振幅放大倍数", 1.0, 3.0, 1.5, step=0.1)

    custom_res = st.checkbox("自定义输出分辨率")
    resolution = None
    if custom_res:
        col1, col2 = st.columns(2)
        width = col1.number_input("宽(px)", 320, 3840, 1920, step=10)
        height = col2.number_input("高(px)", 240, 2160, 1080, step=10)
        resolution = (int(width), int(height))

st.subheader("1️⃣ 选择文件")
audio_file = st.file_uploader("上传音频 (mp3/wav/ogg)", type=["mp3", "wav", "ogg"])
image_file = st.file_uploader("上传背景图 (jpg/png)", type=["jpg", "jpeg", "png"])


def show_preview(audio_path: str, image_path: str):
    bars = prepare_spectrum(audio_path, n_bars, fps)
    bar_vals = bars[fps] if bars.shape[0] > fps else bars[-1]

    bg = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if resolution:
        bg = cv2.resize(bg, resolution, interpolation=cv2.INTER_AREA)
    h, w = bg.shape[:2]
    bg_blur = blur_image(bg, blur_ratio)

    bar_slot_w = w / n_bars
    bar_draw_w = int(bar_slot_w * bar_width_ratio)
    bar_max_h = int(h * bar_height_ratio)
    bottom_offset = int(h * bar_bottom_margin)
    color_bgr = hex_to_bgr(bar_color)

    canvas = bg_blur.copy()
    for i, val in enumerate(bar_vals):
        bar_h = int(min(val * bar_max_h * amp_scale, bar_max_h))
        x1 = int(i * bar_slot_w + (bar_slot_w - bar_draw_w) / 2)
        x2 = x1 + bar_draw_w
        y1 = h - bar_h - bottom_offset
        y2 = h - bottom_offset
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, -1)
    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="实时预览")

if audio_file and image_file:
    st.subheader("2️⃣ 预览 & 试听")
    col_img, col_audio = st.columns(2)
    with col_img:
        st.image(image_file, caption="背景图 (原图)")
    with col_audio:
        st.audio(audio_file)

    if st.button("🔄 更新预览"):
        with st.spinner("生成预览…"):
            tmpdir = tempfile.mkdtemp()
            p_audio = Path(tmpdir) / audio_file.name
            p_img = Path(tmpdir) / image_file.name
            p_audio.write_bytes(audio_file.getbuffer())
            p_img.write_bytes(image_file.getbuffer())
            show_preview(str(p_audio), str(p_img))
            shutil.rmtree(tmpdir, ignore_errors=True)

    st.subheader("3️⃣ 生成视频")
    if st.button("🚀 开始生成", type="primary"):
        with st.spinner("正在渲染，请稍候…"):
            tmpdir = tempfile.mkdtemp()
            p_audio = Path(tmpdir) / audio_file.name
            p_img = Path(tmpdir) / image_file.name
            p_audio.write_bytes(audio_file.getbuffer())
            p_img.write_bytes(image_file.getbuffer())
            out_path = Path(tmpdir) / "output.mp4"

            prog = st.progress(0.0, text="合成中…")

            generate_video(
                str(p_audio),
                str(p_img),
                str(out_path),
                n_bars=n_bars,
                alpha=alpha,
                blur_ratio=blur_ratio,
                fps=fps,
                resolution=resolution,
                bar_color=bar_color,
                bar_height_ratio=bar_height_ratio,
                bar_bottom_margin=bar_bottom_margin,
                bar_width_ratio=bar_width_ratio,
                amp_scale=amp_scale,
                progress_callback=lambda p: prog.progress(p),
            )

        st.success("✅ 视频生成完成！")
        st.video(str(out_path))
        with open(out_path, "rb") as f:
            st.download_button("下载 MP4", f, "visualizer.mp4", "video/mp4")
else:
    st.info("👆 请先上传音频和背景图片")
