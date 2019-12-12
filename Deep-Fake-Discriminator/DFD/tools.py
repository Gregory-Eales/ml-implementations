import torch
import torchvision
import cv2
import numpy as np


def load_mp4_file(path):
    vframe, aframes, info = torchvision.io.read_video(path)
    return vframe, aframes, info


def load_mp4_cv2(path):

    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((10, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < 10  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf


def main():
    p1 = "/Users/gregeales/Desktop/Repositories/ML-Reimplementations/"
    p2 = "Deep-Fake-Discriminator/DFD/sample_vids/vid1.mp4"
    t = load_mp4_cv2(p1+p2)
    t = torch.Tensor(t)
    print(t.shape)



if __name__ == "__main__":
    main()
