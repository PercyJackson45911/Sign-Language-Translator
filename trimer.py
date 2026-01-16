import os
from concurrent.futures import ProcessPoolExecutor
import ffmpeg
import pandas as pd


split = "val"
csv = f"/maindrive/Programing/Python/Sign_Language_Translator/CSV/{split}.csv"
videos = f"/mnt/ML_DATASET/Sign_Language_Translation/how2sign/videos/{split}/"
trimmed_videos = f"/mnt/ML_DATASET/Sign_Language_Translation/how2sign/trimmed/{split}/"
failed_txt = (
    f"/maindrive/Programing/Python/Sign_Language_Translator/failed/{split}_failed.txt"
)
metadata = pd.read_csv(csv, sep="\t")


def failed(error_message):
    with open(failed_txt, "a") as f:
        f.write(error_message + "\n")


def trim_video(y):
    start_time = y["START_REALIGNED"]
    end_time = y["END_REALIGNED"]
    duration = end_time - start_time
    video_id = y["VIDEO_NAME"]
    save_name = y["SENTENCE_NAME"]
    video_filepath = f"{videos}{video_id}.mp4"
    save_filepath = f"{trimmed_videos}{save_name}.mp4"
    if not os.path.exists(video_filepath):
        failed(f"{video_filepath} doesn't exist")
        return
    if os.path.exists(save_filepath):
        return
    input_video = ffmpeg.input(video_filepath, ss=start_time)
    output_video = ffmpeg.output(
        input_video,
        save_filepath,
        t=duration,
        vcodec="libx264",  # ensures proper video stream
        an=None,  # optional, or acodec='none' if no audio needed
        strict="experimental",
        movflags="+faststart",  # fixes header for streaming / OpenCV
    )
    ffmpeg.run(output_video, overwrite_output=True)


with ProcessPoolExecutor(max_workers=2) as executor:
    list(executor.map(trim_video, [row for _, row in metadata.iterrows()]))
    list(executor.map(trim_video, [row for _, row in metadata.iterrows()]))
