from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import os
import shutil
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",                                          # local dev
        "http://localhost:8080",                                          # local alt port
        "https://ai-exam-frontend-production.up.railway.app",            # Railway frontend
        "*"                                                               # fallback (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    original_name = file.filename or f"video_{uuid.uuid4()}.mp4"
    file_path = os.path.join(UPLOAD_FOLDER, original_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        os.remove(file_path)
        return {"error": "Could not open video file"}

    emotion_counts = {
        "happy": 0, "neutral": 0, "sad": 0,
        "angry": 0, "fear": 0, "disgust": 0, "surprise": 0
    }
    frame_count = 0
    total_analyzed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = result[0]["dominant_emotion"]
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                total_analyzed += 1
            except:
                pass

    cap.release()

    if os.path.exists(file_path):
        os.remove(file_path)

    dominant = max(emotion_counts, key=emotion_counts.get) if total_analyzed > 0 else "neutral"

    return {
        "video": original_name,
        "dominant_emotion": dominant,
        "total_analyzed_frames": total_analyzed,
        "happy_frames": emotion_counts["happy"],
        "neutral_frames": emotion_counts["neutral"],
        "sad_frames": emotion_counts["sad"],
        "angry_frames": emotion_counts["angry"],
        "fear_frames": emotion_counts["fear"],
        "disgust_frames": emotion_counts["disgust"],
        "surprise_frames": emotion_counts["surprise"]
    }

@app.get("/health")
def health():
    return {"status": "Face Analysis Service is running"}
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from deepface import DeepFace
# import cv2
# import os
# import shutil
# import uuid

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_FOLDER = r"C:\Users\HP\Documents\IPMsg\AutoSave"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.post("/analyze-video")
# async def analyze_video(file: UploadFile = File(...)):

#     original_name = file.filename or f"video_{uuid.uuid4()}.mp4"
#     file_path = os.path.join(UPLOAD_FOLDER, original_name)

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     cap = cv2.VideoCapture(file_path)
#     if not cap.isOpened():
#         return {"error": "Could not open video file"}

#     emotion_counts = {
#         "happy": 0, "neutral": 0, "sad": 0,
#         "angry": 0, "fear": 0, "disgust": 0, "surprise": 0
#     }
#     frame_count = 0
#     total_analyzed = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         if frame_count % 30 == 0:
#             try:
#                 result = DeepFace.analyze(
#                     frame,
#                     actions=['emotion'],
#                     enforce_detection=False
#                 )
#                 emotion = result[0]["dominant_emotion"]
#                 if emotion in emotion_counts:
#                     emotion_counts[emotion] += 1
#                 total_analyzed += 1
#             except:
#                 pass

#     cap.release()

#     # Dominant emotion
#     dominant = max(emotion_counts, key=emotion_counts.get) if total_analyzed > 0 else "neutral"

#     # Return ONLY raw counts — frontend calculates all scores
#     return {
#         "video": original_name,
#         "dominant_emotion": dominant,
#         "total_analyzed_frames": total_analyzed,
#         "happy_frames": emotion_counts["happy"],
#         "neutral_frames": emotion_counts["neutral"],
#         "sad_frames": emotion_counts["sad"],
#         "angry_frames": emotion_counts["angry"],
#         "fear_frames": emotion_counts["fear"],
#         "disgust_frames": emotion_counts["disgust"],
#         "surprise_frames": emotion_counts["surprise"]
#     }