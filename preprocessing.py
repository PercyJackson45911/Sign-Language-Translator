import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor



# file paths and other global unchanging variables here
split = "train"
csv = f"/maindrive/Programing/Python/Sign_Language_Translator/CSV/{split}.csv"
videos = f"/mnt/ML_DATASET/Sign_Language_Translation/how2sign/trimmed/{split}/"
destination = f"/mnt/ML_DATASET/Sign_Language_Translation/how2sign/pt_files/{split}/"
failed_txt = f"/maindrive/Programing/Python/Sign_Language_Translator/failed/{split}_failed.txt"
hand_model = "hand_tracking.task"
face_model = "face_landmarker.task"
metadata = pd.read_csv(csv, sep="\t")

# Mediapipe Options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model),
    running_mode=VisionRunningMode.VIDEO,
)
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model),
    running_mode=VisionRunningMode.VIDEO,
)

def failed(error_message):
    with open(failed_txt, "a") as f:
        f.write(error_message + "\n")

def preprocessor(y):
    video_id = y["SENTENCE_NAME"]
    sentence = y["SENTENCE"]
    video_filepath = f"{videos}{video_id}.mp4"
    pt_files = f"{destination}{video_id}.pt"
    vid = cv2.VideoCapture(video_filepath)
    fps = vid.get (cv2.CAP_PROP_FPS)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_landmarks_array = np.zeros((length ,2, 21, 3), dtype=np.float16)
    face_landmarks_array = np.zeros((length, 1,468,3), dtype=np.float16)
    if os.path.exists(pt_files):
        return  # skip if pt file already exists
    if not os.path.exists(video_filepath):
        failed(f"{video_filepath} doesn't exist") # lets us know if any video files are missing
        return
    # these are the corruption checks to make sure the video itself has data
    if fps == 0:
        failed(
            f"{video_filepath} is corrupted and doesnt have a valid fps.. please check source and either rectify or yeet from csv"
        )
        return
    if length == 0:
        failed(
            f'{video_filepath} is corrupted but this time instead of fps, it aint got any frames itself.. anyways check source and rectify or yeetus deletus'
        )
        return
    # now we initialize Handlandmarker.. before you think of initializing once.. this is video mode. You'd have temporal bias bleed
    with (
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        FaceLandmarker.create_from_options(face_options) as face_landmarker
    ):
        framecount = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            else:
                framecount += 1
                timestamp = int((framecount / fps) * 1000)
                mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                hand_result = hand_landmarker.detect_for_video(mp_Image, timestamp)
                face_result = face_landmarker.detect_for_video(mp_Image, timestamp)
                for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness): # we get hte
                    if handedness[0].category_name == 'Left':
                        hand_index=0
                    else:
                        hand_index=1
                    for j, lm in enumerate(hand_landmarks[:21]):
                        hand_landmarks_array[framecount-1, hand_index, j] = [lm.x, lm.y, lm.z]
                for face_landmarks in face_result.face_landmarks:
                    face_landmarks_array[framecount-1, 0] = [[lm.x, lm.y, lm.z] for lm in face_landmarks]

            if abs((framecount/timestamp)*1000 - fps)>2:
                failed(
                    f'{video_filepath} may be corrupt. calculated FPS was {framecount/timestamp} and the actual FPS of the video was {fps}'
                       )
    vid.release()
    if np.all(hand_landmarks_array == 0):
        failed(f"{video_filepath} has no landmarks")
        return
    hand_tensor = torch.tensor(hand_landmarks_array, dtype=torch.float16)
    face_tensor = torch.tensor(face_landmarks_array, dtype=torch.float16)
    data_dict = {"gloss": sentence, "hand_landmarks": hand_tensor, "face_landmarks": face_tensor}
    torch.save(data_dict, f"{destination}{video_id}.pt")

with ProcessPoolExecutor(max_workers=2) as executor:
    list(executor.map(preprocessor, [row for _, row in metadata.iterrows()]))
