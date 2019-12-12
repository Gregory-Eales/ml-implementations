from DFD.tools import load_mp4_cv2
import os
from os import listdir
from os.path import isfile, join
import torch
from tqdm import tqdm


cwd = os.getcwd()
path = cwd + "\\Deep-Fake-Discriminator\\DFD\\Deep-Fake-Data\\train_sample_videos"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

path2 = "Deep-Fake-Discriminator/DFD/torch_data/"
for i in tqdm(range(len(onlyfiles))):
    vid = load_mp4_cv2(path+"//"+onlyfiles[i])
    torch.save(vid, path2+onlyfiles[i][0:-4]+ ".pt")
