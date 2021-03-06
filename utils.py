import os
import shutil

import numpy as np


def get_name_and_ext(path):
    name, ext = os.path.splitext(os.path.basename(path))
    return name, ext


def add_noise_for_waveform(s, n, db):
    """
    为语音文件叠加噪声
    ----
    para:
        s：原语音的时域信号
        n：噪声的时域信号
        db：信噪比
    ----
    return:
        叠加噪声后的语音
    """
    alpha = np.sqrt(
        np.sum(s ** 2) / (np.sum(n ** 2) * 10 ** (db / 10))
    )
    mix = s + alpha * n
    return mix


def prepare_empty_dirs(dirs: list):
    for d in dirs:
        if d.exists():
            shutil.rmtree(d.as_posix())
        d.mkdir(parents=True, exist_ok=False)
