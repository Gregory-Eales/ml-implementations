import torch
import torchvision

def load_mp4_file(path):
    vframe, aframes, info = torchvision.io.read_video(path)
    return vframe, aframes, info
