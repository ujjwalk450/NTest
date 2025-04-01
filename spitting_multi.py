import os
import time
import psutil
import subprocess
import multiprocessing
from glob import glob

VIDEO_DIR = "videos"
OUTPUT_DIR = "results"
FFMPEG_RES = "640x360"
BATCH_SIZE = 16
MODEL_WEIGHTS = "spitting_detector_weights_48.h5"
POSE_MODEL = "pose_landmarker_full.task"

def get_dynamic_worker_count():
    total_cores = multiprocessing.cpu_count()
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    if ram > 75 or cpu > 85:
        return 2
    elif ram > 60:
        return max(5, total_cores)
    return min(10, total_cores)

def downscale_video(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale={FFMPEG_RES}",
        "-preset", "ultrafast", "-loglevel", "error", output_path
    ]
    subprocess.run(cmd, check=True)

def process_single_video(video_path):
    import cv2
    import numpy as np
    import math
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

    def predict_spitting(model, roi):
        roi = roi.astype("float32") / 255.0
        prediction = model.predict(roi, verbose=0)[0][0]
        return prediction > 0.75

    def get_point(landmarks, idx, w, h):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

    def angle_between(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return (math.degrees(math.atan2(-dy, dx)) + 360) % 360

    base = os.path.basename(video_path)
    video_name = os.path.splitext(base)[0]
    save_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(save_dir, exist_ok=True)
    downscaled_path = os.path.join(save_dir, f"{video_name}_360p.mp4")
    downscale_video(video_path, downscaled_path)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), Dropout(0.25),
        Flatten(), Dense(512, activation='relu'),
        Dropout(0.5), Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights(MODEL_WEIGHTS)

    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=4,
            min_pose_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
    )

    cap = cv2.VideoCapture(downscaled_path)
    frame_count = 0
    detected = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = pose_landmarker.detect_for_video(mp_image, frame_count * 33)

        for idx, pose in enumerate(results.pose_landmarks):
            try:
                pose_id = results.pose_world_landmarks[idx].pose_id
            except AttributeError:
                pose_id = idx

            if pose_id in detected:
                continue

            l_sh = get_point(pose, 11, w, h)
            r_sh = get_point(pose, 12, w, h)
            l_ear = get_point(pose, 7, w, h)
            r_ear = get_point(pose, 8, w, h)

            mid_sh = ((l_sh + r_sh) / 2).astype(int)
            mid_ear = ((l_ear + r_ear) / 2).astype(int)

            tilt = int(angle_between(mid_sh, mid_ear) - 90)
            shoulder = int(angle_between(l_sh, r_sh))

            match = (tilt > 10 and 10 < shoulder < 30) or (tilt < -10 and shoulder < 350)
            if not match:
                continue

            center = ((mid_sh + mid_ear) / 2).astype(int)
            width = int(np.linalg.norm(r_sh - l_sh))
            x1 = max(0, center[0] - width)
            y1 = max(0, center[1] - width)
            x2 = min(w, center[0] + width)
            y2 = min(h, center[1] + width)

            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = np.expand_dims(np.expand_dims(gray, -1), 0)

            if predict_spitting(model, gray):
                detected.add(pose_id)
                fname = os.path.join(save_dir, f"spitting_{frame_count}_id{pose_id}.jpg")
                cv2.imwrite(fname, roi)
    cap.release()

# === MAIN CONTROLLER ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_list = glob(os.path.join(VIDEO_DIR, "*.mp4"))
    total = len(video_list)
    start_time = time.time()
    print(start_time)
    for i in range(0, total, BATCH_SIZE):
        batch = video_list[i:i+BATCH_SIZE]
        workers = get_dynamic_worker_count()
        batch_start = time.time()
        print(f"\n[Batch {i//BATCH_SIZE + 1}] Processing {len(batch)} videos with {workers} worker(s)...")

        with multiprocessing.Pool(processes=workers) as pool:
            pool.map(process_single_video, batch)

        elapsed = time.time() - batch_start
        processed = i + len(batch)
        remaining = total - processed
        est_time = (elapsed / len(batch)) * remaining if remaining else 0
        print(f"[✓] Batch processed in {elapsed:.1f}s — {processed}/{total} done")
        print(f"[⏳] Estimated time remaining: {int(est_time)}s")

    print(f"\n✅ All {total} videos processed in {int(time.time() - start_time)} seconds.")
