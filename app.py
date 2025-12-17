import streamlit as st
import numpy as np
import os
import uuid

from DJriks import (
    analyze_library,
    compare_songs,
    extract_features,
    detect_genre,
    MUSIC_DIR,
    UPLOAD_DIR
)

# ===== Streamlit konfigurācija =====
st.set_page_config(
    page_title="DJ Similarity App",
    layout="wide"
)

st.title("🎧 DJ dziesmu līdzības meklētājs")

# ===== Augšupielādes mape (Streamlit Cloud droša) =====
library_path = MUSIC_DIR          # statiskās dziesmas no GitHub
upload_path = UPLOAD_DIR          # /tmp/uploads (pagaidu)
os.makedirs(upload_path, exist_ok=True)

# ===== Augšupielāde =====
uploaded = st.file_uploader("⬆️ Augšupielādē dziesmu", type=["mp3", "wav"])

uploaded_name = None
if uploaded:
    uploaded_name = f"{uuid.uuid4()}.mp3"
    upload_file_path = os.path.join(upload_path, uploaded_name)

    with open(upload_file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("Dziesma augšupielādēta")

# ===== Ielādē bibliotēku no DJriks =====
features = analyze_library()

# ===== Pievieno augšupielādēto dziesmu =====
if uploaded_name:
    feat = extract_features(upload_file_path)
    feat["genre"] = detect_genre(feat)
    features[uploaded_name] = feat

files = list(features.keys())

if not files:
    st.warning("Bibliotēkā nav dziesmu")
    st.stop()

# ===== Enerģijas normalizācija 0–100% =====
energies = [feat["energy"] for feat in features.values()]
min_e, max_e = min(energies), max(energies)

for f in features:
    e = features[f]["energy"]
    if max_e - min_e > 0:
        features[f]["energy"] = 100 * (e - min_e) / (max_e - min_e)
    else:
        features[f]["energy"] = 50.0

# ===== DZIESMU SARAKSTS =====
st.subheader("🎵 Pieejamās dziesmas")

for f in files:
    c1, c2 = st.columns([4, 1])
    c1.write(
        f"{f} — {features[f]['genre']} | "
        f"BPM {features[f]['tempo']:.1f} | "
        f"Enerģija {features[f]['energy']:.1f}%"
    )

    audio_path = (
        os.path.join(upload_path, f)
        if f in os.listdir(upload_path)
        else os.path.join(library_path, f)
    )
    c2.audio(audio_path)

# ===== SĀKUMA DZIESMAS IZVĒLE =====
choice = st.selectbox("🎶 Izvēlies sākuma dziesmu:", files)
input_feat = features[choice]

st.subheader("🔊 Izvēlētā dziesma")
st.write(
    f"{choice} — {input_feat['genre']} | "
    f"BPM {input_feat['tempo']:.1f} | "
    f"Enerģija {input_feat['energy']:.1f}%"
)

choice_path = (
    os.path.join(upload_path, choice)
    if choice in os.listdir(upload_path)
    else os.path.join(library_path, choice)
)
st.audio(choice_path)

# ===== SALĪDZINĀŠANAS PARAMETRS =====
param = st.selectbox(
    "Salīdzināt pēc:",
    ["BPM", "Enerģija", "MFCC", "Viss kopā", "Bungas / Ritms", "Žanrs"]
)

# ===== SALĪDZINĀŠANA =====
if param == "BPM":
    base = input_feat["tempo"]
    res = sorted(
        [(f, abs(base - features[f]["tempo"])) for f in files if f != choice],
        key=lambda x: x[1]
    )

    st.subheader("📊 Līdzīgākās dziesmas pēc BPM")
    for f, d in res[:5]:
        st.write(f"{f} — Δ {d:.1f}")
        audio_path = (
            os.path.join(upload_path, f)
            if f in os.listdir(upload_path)
            else os.path.join(library_path, f)
        )
        st.audio(audio_path)

elif param == "Enerģija":
    input_energy = input_feat["energy"]
    res = []

    for f in files:
        if f == choice:
            continue
        sim = 1 - abs(input_energy - features[f]["energy"]) / 100
        res.append((f, sim))

    res = sorted(res, key=lambda x: x[1], reverse=True)

    st.subheader("🎵 Līdzīgākās dziesmas pēc Enerģijas")
    for f, s in res[:5]:
        st.write(f"{f} — Līdzība: {s*100:.1f}%")
        audio_path = (
            os.path.join(upload_path, f)
            if f in os.listdir(upload_path)
            else os.path.join(library_path, f)
        )
        st.audio(audio_path)

else:
    # MFCC / Viss kopā / Bungas / Ritms / Žanrs
    results = compare_songs(features, choice, param)

    st.subheader("🎵 Līdzīgākās dziesmas")
    for f, s in results[:5]:
        st.write(f"{f} — Līdzība: {s*100:.1f}%")
        audio_path = (
            os.path.join(upload_path, f)
            if f in os.listdir(upload_path)
            else os.path.join(library_path, f)
        )
        st.audio(audio_path)
