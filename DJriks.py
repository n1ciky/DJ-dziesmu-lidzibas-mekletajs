import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.join(BASE_DIR, "music_library")
UPLOAD_DIR = "/tmp/uploads"

os.makedirs(MUSIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------- Feature extraction --------
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True)

    try:
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    except:
        zcr, centroid = 0.0, 0.0

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except:
        tempo = 0.0

    try:

        
        energy = float(np.mean(librosa.feature.rms(y=y)))
    except:
        energy = 0.0

    try:
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    except:
        mfcc = np.zeros(13)

    return {
        "tempo": tempo,
        "energy": energy,
        "mfcc": mfcc,
        "zcr": zcr,
        "centroid": centroid
    }

# -------- Genre --------
def detect_genre(feat):
    if feat["tempo"] > 130 and feat["energy"] > 0.03:
        return "Electronic / Club"
    elif feat["tempo"] < 90 and feat["energy"] < 0.02:
        return "Chill / Hip-hop"
    elif 90 <= feat["tempo"] <= 120:
        return "Pop / Indie"
    return "Other"

# -------- Analyze library --------
def analyze_library():
    files = [f for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]
    features = {}

    for f in files:
        feat = extract_features(os.path.join(MUSIC_DIR, f))
        feat["genre"] = detect_genre(feat)
        features[f] = feat

    return features

# -------- Similarity --------
def compare_songs(features, base_song, mode):
    base_feat = features[base_song]
    vecs, names = [], []

    for name, feat in features.items():
        if name == base_song:
            continue

        if mode == "MFCC":
            vec = feat["mfcc"]
            base_vec = base_feat["mfcc"]
        elif mode == "Viss kopā":
            vec = np.concatenate(([feat["tempo"], feat["energy"]], feat["mfcc"]))
            base_vec = np.concatenate(([base_feat["tempo"], base_feat["energy"]], base_feat["mfcc"]))
        elif mode == "Bungas / Ritms":
            vec = [feat["zcr"], feat["centroid"]]
            base_vec = [base_feat["zcr"], base_feat["centroid"]]
        else:
            continue

        vecs.append(vec)
        names.append(name)

    scaler = StandardScaler()
    X = scaler.fit_transform(vecs)
    q = scaler.transform([base_vec])

    scores = (cosine_similarity(q, X)[0] + 1) / 2
    return sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
