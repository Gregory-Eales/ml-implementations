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

def get_training_sample_paths():
    path = '../input/deepfake-detection-challenge/train_sample_videos'
    paths = os.listdir(path)
    return paths

def load_training_samples(a, b, n_frames):
    path = '../input/deepfake-detection-challenge/train_sample_videos'
    paths = get_training_sample_paths()
    training_samples = []
    for p in paths[a:b]:
        if p[-3:] == "mp4":
            training_samples.append(load_video(path+"/"+p, n_frames=n_frames))
    return training_samples

def load_video(path, n_frames=None):
    cap = cv2.VideoCapture(path)

    # use if selected a certain number of frames or entire video
    if n_frames == None:
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else: frameCount = n_frames

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf

def load_training_sample_meta():
    path = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'
    df_meta = pd.read_json(path)
    return df_meta


def generate_data(n_frames, n_vids, vid_indx):
    df_meta = load_training_sample_meta()
    df_meta.loc['label'] = df_meta.loc['label'].map({'FAKE': 1, 'REAL': 0})
    path = '../input/deepfake-detection-challenge/train_sample_videos/'
    vid_files = df_meta.columns
    print(len(vid_files))
    x = []
    y = []
    for i in tqdm(range(vid_indx, vid_indx+n_vids)):
        x.append(load_video(path+vid_files[i], n_frames=n_frames).reshape([3, 10, 1080, 1920]))
        y.append(df_meta[vid_files[i]]['label'])

    x = np.stack(x)
    y = np.array(y).reshape([len(y), 1])

    return torch.Tensor(x).float(), torch.Tensor(y).float()

def main():
    p1 = "C:/Users/Greg/Desktop/Repositories/ML-Reimplementations"
    p2 = "/Deep-Fake-Discriminator/DFD/sample_vids/vid1.mp4"
    t = load_mp4_cv2(p1+p2)
    t = torch.Tensor(t)
    print(t.shape)

if __name__ == "__main__":
    main()
