import torch
import skvideo.io
import cv2
import numpy as np


def load_mp4_file(path):
    return skvideo.io.vread(path)

def load_mp4_cv2(path):

    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    buf = torch.Tensor(buf).type(torch.uint8)
    return buf

def main():
    p1 = "C:/Users/Greg/Desktop/Repositories/ML-Reimplementations"
    p2 = "/Deep-Fake-Discriminator/DFD/sample_vids/vid1.mp4"
    t = load_mp4_cv2(p1+p2)
    t = torch.Tensor(t)
    print(t.shape)
    
if __name__ == "__main__":
    main()
