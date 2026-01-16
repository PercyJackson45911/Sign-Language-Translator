import os
import concurrent.futures as multithread
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch


# file paths and other global unchanging variables here
split = "train"
csv = f"/maindrive/Programing/Python/Sign_ Language_Translator/CSV/{split}.csv"
videos = f"/mnt/ML_DATASET/Sign_Language_Translation/how2sign/trimmed/{split}/"
destination = f"/mnt/ML_DATASET/Sign_Language_Translation/how2sign/pt_files/{split}/"
failed_txt = f"/maindrive/Programing/Python/Sign_Language_Translator/failed/{split}_failed.txt"
hand_model = "hand_tracking.task"
metadata = pd.read_csv(csv, sep="\t")

def failed(error_message):
    with open(failed_txt, "a") as f:
        f.write(error_message + "\n")

def preprocessor(y):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        running_mode=VisionRunningMode.VIDEO,
    )
    landmarks = []

    landmarks.clear()
    video_id = y["SENTENCE_NAME"]
    sentence = y["SENTENCE"]
    video_filepath = f"{videos}{video_id}.mp4"
    pt_files = f"{destination}{video_id}.pt"
    if os.path.exists(pt_files):
        return  # skip if pt file already exists
    if not os.path.exists(video_filepath):
        failed(f"{video_filepath} doesn't exist")
        return
    vid = cv2.VideoCapture(video_filepath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        failed(
            f"{video_filepath} is corrupted.. please check source and either rectify or yeet from csv"
        )
        return
    with HandLandmarker.create_from_options(options) as landmarker:
        framecount = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                failed(f"{video_filepath} can't be read by opencv2")
                break
            else:
                framecount += 1
                timestamp = int((framecount / fps) * 1000)
                mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = landmarker.detect_for_video(mp_Image, timestamp)
                frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)
                for i, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                    for j, lm in enumerate(hand_landmarks[:21]):  # max 21 landmarks
                        frame_landmarks[i, j] = [lm.x, lm.y, lm.z]
                landmarks.append(frame_landmarks)
    vid.release()
    landmarks_array = np.array(landmarks)
    if np.all(landmarks_array == 0):
        failed(f"{video_filepath} has no landmarks")
        return
    tensor = torch.tensor(landmarks_array, dtype=torch.float32)
    data_dict = {"gloss": sentence, "landmarks": tensor}
    torch.save(data_dict, f"{destination}{video_id}.pt")


with multithread(max_workers=2) as executor:
    list(executor.map(preprocessor), [row for _, row in metadata.iterrows()])

