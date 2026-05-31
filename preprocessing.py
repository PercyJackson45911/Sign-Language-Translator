import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import gc


# file paths and other global unchanging variables here
split = "train"
csv = f"/maindrive/Programing/ASL_Citizen/splits/{split}.csv"
videos = f"/maindrive/Programing/ASL_Citizen/videos/"
destination = f"/maindrive/Programing/ASL_Citizen/ptfs/{split}/"
failed_txt = f"/maindrive/Programing/Sign-Language-Translator/failed/{split}_failed.txt"
hand_model = "hand_landmarker.task"
face_model = "face_landmarker.task"
pose_model = "pose_landmarker.task"  # were using the light version here.. feel free to use others
metadata = pd.read_csv(csv, sep=",")


def failed(error_message):
    with open(failed_txt, "a") as f:
        f.write(error_message + "\n")

with open(failed_txt) as f:
    content = f.read()

def preprocessor(y):
    # Mediapipe Options
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model),
        running_mode=VisionRunningMode.VIDEO,
    )
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model),
        running_mode=VisionRunningMode.VIDEO,
    )

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=VisionRunningMode.VIDEO,
    )

    video_id = y["Video file"]
    sentence = y["Gloss"]
    video_filepath = f"{videos}{video_id}"
    pt_files = f"{destination}{video_id}.pt"
    vid = cv2.VideoCapture(video_filepath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


    try:
        if os.path.exists(pt_files):
            return  # skip if pt file already exists
        if video_filepath in content:
            return
        if not os.path.exists(video_filepath):
            failed(
                f"{video_filepath} doesn't exist"
            )  # lets us know if any video files are missing
            return
        # these are the corruption checks to make sure the video itself has data
        if fps == 0:
            failed(
                f"{video_filepath} is corrupted and doesnt have a valid fps.. please check source and either rectify or yeet from csv"
            )
            return
        if length == 0:
            failed(
                f"{video_filepath} is corrupted but this time instead of fps, it aint got any frames itself.. anyways check source and rectify or yeetus deletus"
            )
            return
        # now we initialize Handlandmarker.. before you think of initializing once.. this is video mode. You'd have temporal bias bleed
        with (
            HandLandmarker.create_from_options(hand_options) as hand_landmarker,
            FaceLandmarker.create_from_options(face_options) as face_landmarker,
            PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        ):
            framecount = 0
            hand_landmarks_array = np.zeros((length, 2, 21, 3), dtype=np.float16)
            face_landmarks_array = np.zeros((length, 1, 478, 3), dtype=np.float16)
            pose_landmarks_array = np.zeros((length, 1, 33, 3), dtype=np.float16)
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
                    pose_result = pose_landmarker.detect_for_video(mp_Image, timestamp)


                    for hand_landmarks, handedness in zip(
                            hand_result.hand_landmarks, hand_result.handedness
                    ):  # we get hte
                        if handedness[0].category_name == "Left":
                            hand_index = 0
                        else:
                            hand_index = 1
                        for j, lm in enumerate(hand_landmarks[:21]):
                            hand_landmarks_array[framecount - 1, hand_index, j] = [
                                lm.x,
                                lm.y,
                                lm.z,
                            ]
                    for face_landmarks in face_result.face_landmarks:
                        face_landmarks_array[framecount - 1, 0] = [
                            [lm.x, lm.y, lm.z] for lm in face_landmarks
                        ]
                    for pose_landmarks in pose_result.pose_landmarks:
                        pose_landmarks_array[framecount - 1, 0] = [
                            [lm.x, lm.y, lm.z] for lm in pose_landmarks
                        ]
        del mp_Image, frame, face_landmarker, hand_landmarker, pose_landmarker

        if np.all(hand_landmarks_array == 0):
            failed(f"{video_filepath} has no landmarks")
            return
        hand_tensor = torch.from_numpy(hand_landmarks_array, dtype=torch.float16)
        face_tensor = torch.from_numpy(face_landmarks_array, dtype=torch.float16)
        pose_tensor = torch.from_numpy(pose_landmarks_array, dtype=torch.float16)
        del hand_landmarks_array, face_landmarks_array, pose_landmarks_array
        data_dict = {
            "gloss": sentence,
            "hand_landmarks": hand_tensor,
            "face_landmarks": face_tensor,
            "pose_landmarks": pose_tensor,
        }
        torch.save(data_dict, f"{destination}{video_id}.pt")

    finally:
        vid.release()
        gc.collect()


if __name__ == "__main__":
    print("starting", flush=True)
    rows = [row for _, row in metadata.iterrows()]
    with ProcessPoolExecutor(max_workers=12, max_tasks_per_child = 1) as executor:
        futures = [executor.submit(preprocessor, row) for row in rows]
        for i, _ in enumerate(as_completed(futures), 1):
            print(f"\r{i}/{len(futures)}", end="", flush=True, file=sys.stdout)
