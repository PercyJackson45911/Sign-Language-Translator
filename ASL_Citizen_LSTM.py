import torch
import pandas as pd

df = pd.read_csv('/maindrive/Programing/ASL_Citizen/splits/test.csv')

print(torch.load(f'/maindrive/Programing/ASL_Citizen/ptf/test/{df.loc[0, "Video file"]}.pt')['face_landmarks'].shape)
print(torch.load(f'/maindrive/Programing/ASL_Citizen/ptf/test/{df.loc[0, "Video file"]}.pt')['pose_landmarks'].shape)
print(torch.load(f'/maindrive/Programing/ASL_Citizen/ptf/test/{df.loc[0, "Video file"]}.pt')['hand_landmarks'].shape)