from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import os
import shutil
import uuid
import psycopg2
from urllib.parse import urlparse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:8080",
        "https://ai-exam-frontend-production.up.railway.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ PostgreSQL connection using DATABASE_URL
def get_db():
    database_url = os.getenv("DATABASE_URL")
    url = urlparse(database_url)
    return psycopg2.connect(
        host=url.hostname,
        port=url.port,
        database=url.path[1:],
        user=url.username,
        password=url.password
    )

# ✅ Create table on startup
@app.on_event("startup")
def create_table():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_analysis (
                id SERIAL PRIMARY KEY,
                video_name VARCHAR(255),
                dominant_emotion VARCHAR(50),
                total_analyzed_frames INT,
                happy_frames INT,
                neutral_frames INT,
                sad_frames INT,
                angry_frames INT,
                fear_frames INT,
                disgust_frames INT,
                surprise_frames INT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Table created successfully")
    except Exception as e:
        print(f"❌ Table creation error: {e}")

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

    # ✅ Save to PostgreSQL
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO video_analysis (
                video_name, dominant_emotion, total_analyzed_frames,
                happy_frames, neutral_frames, sad_frames,
                angry_frames, fear_frames, disgust_frames, surprise_frames
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            original_name, dominant, total_analyzed,
            emotion_counts["happy"], emotion_counts["neutral"], emotion_counts["sad"],
            emotion_counts["angry"], emotion_counts["fear"],
            emotion_counts["disgust"], emotion_counts["surprise"]
        ))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Data saved for video: {original_name}")
    except Exception as e:
        print(f"❌ DB Error: {e}")

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
