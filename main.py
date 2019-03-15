import glob
import os
import random
import shutil
from pathlib import Path

import librosa
from tqdm import tqdm

from utils import get_name_and_ext, add_noise_for_waveform


"""============ Begin Config ============"""
# 训练集中 wav 文件的数量
# NUM_OF_TRAIN_WAV_FILES = 400
NUM_OF_TRAIN_WAV_FILES = 800

# 测试集的 wav 文件的数量（包含训练集中的 wav 文件）
# NUM_OF_TEST_WAV_FILES = 500
NUM_OF_TEST_WAV_FILES = 1000

# 用于训练的信噪比集合
DBS_TRAIN = ["-15", "-10" ,"-5", "0"]
# DBS_TRAIN = ["-5", "-10"]

# 用于测试的信噪比集合
DBS_TEST = ["-20", "-17", "-15", "-12", "-10", "-7", "-5", "3", "0", "5", "-3"]

# 用于训练的噪声类型集合
NOISE_TYPE_TRAIN = ["babble", "destroyerengine", "destroyerops", "factoryfloor1"]

# 用于测试的噪声类型集合
NOISE_TYPE_TEST = ["babble", "destroyerengine", "destroyerops", "factoryfloor1", "factoryfloor2"]
"""============ End Config ============"""


# DataStore
DATA_PATH = Path("data")
CLEAN_DATA_PATH = DATA_PATH / "clean"
NOISE_DATA_PATH = DATA_PATH / "noise"

# release dir
RELEASE_DIR = Path("release")
RELEASE_DIR_FOR_TRAIN_CLEAN = RELEASE_DIR / "train" / "clean"
RELEASE_DIR_FOR_TRAIN_NOISY = RELEASE_DIR / "train" / "noisy"
RELEASE_DIR_FOR_TEST_CLEAN = RELEASE_DIR / "test" / "clean"
RELEASE_DIR_FOR_TEST_NOISY = RELEASE_DIR / "test" / "noisy"

# 清空现有的所有相关目录
for dir in [RELEASE_DIR_FOR_TRAIN_CLEAN, RELEASE_DIR_FOR_TRAIN_NOISY, RELEASE_DIR_FOR_TEST_CLEAN, RELEASE_DIR_FOR_TEST_NOISY]:
    if dir.exists():
        shutil.rmtree(dir.as_posix())
    dir.mkdir(parents=True, exist_ok=False)

clean_wav_path = librosa.util.find_files(CLEAN_DATA_PATH.as_posix(), ext=["WAV"], recurse=True)
random.shuffle(clean_wav_path) # select wav file randomly
train_clean_wav_paths = clean_wav_path[:NUM_OF_TRAIN_WAV_FILES]
test_clean_wav_paths = clean_wav_path[:NUM_OF_TEST_WAV_FILES]

# noise wav file path
noise_wav_list = [p for p in glob.glob(NOISE_DATA_PATH.as_posix() + "/*.wav")]

train_noise_wav_paths = [p for p in noise_wav_list if get_name_and_ext(p)[0] in NOISE_TYPE_TRAIN]
test_noise_wav_paths = [p for p in noise_wav_list if get_name_and_ext(p)[0] in NOISE_TYPE_TEST]


def load_wavs(file_paths, sr=16000):
    wavs = []
    for fp in tqdm(file_paths, desc="Loading wavs: "):
        wav, _ = librosa.load(fp, sr=sr)
        wavs.append(wav)
    return wavs


def load_noises(noise_wav_paths):
    """
    加载噪声文件
    Args:
        noise_wav_paths (list): 噪声文件的路径列表

    Returns:
        dict: {"babble": [signals]}
    """
    out = {}
    for noise_path in tqdm(noise_wav_paths, desc="Loading Noises: "):
        name, _ = get_name_and_ext(noise_path)
        wav, _ = librosa.load(noise_path, sr=16000)
        out[name] = wav

    return out


def add_noise_for_full_wavs():
    """
    为所有 wavs 叠加噪声
    Returns:
    Notes:
        测试集合是训练集合的父集合
    """
    noises_obj = load_noises(test_noise_wav_paths)
    clean_wavs = load_wavs(test_clean_wav_paths)

    for num, clean_wav in tqdm(enumerate(clean_wavs), desc="Iteration of all wavs"):
        for noise_name in noises_obj.keys():
            for dB in DBS_TEST:
                output_wav_filename = "{num}_{noise_name}_{dB}.wav".format(
                    num=str(num + 1).zfill(4),
                    noise_name=noise_name,
                    dB=dB
                )
                output_clean_p = os.path.join(RELEASE_DIR_FOR_TEST_CLEAN, output_wav_filename)
                output_noisy_p = os.path.join(RELEASE_DIR_FOR_TEST_NOISY, output_wav_filename)

                mix_wav = add_noise_for_waveform(clean_wav, noises_obj[noise_name][:len(clean_wav)], int(dB))

                assert len(mix_wav) == len(clean_wav)
                librosa.output.write_wav(output_clean_p, clean_wav, 16000)
                librosa.output.write_wav(output_noisy_p, mix_wav, 16000)

def release_data():
    """
    从测试集合中选出需要的训练集
    """
    add_noise_for_full_wavs()

    paths_of_all_clean_wav = sorted(glob.glob(RELEASE_DIR_FOR_TEST_CLEAN.as_posix() + "/*.wav"))
    paths_of_all_noisy_wav = sorted(glob.glob(RELEASE_DIR_FOR_TEST_NOISY.as_posix() + "/*.wav"))

    target_fnames = ["{num}_{noise_name}_{dB}".format(
        num=str(num + 1).zfill(4),
        noise_name=noise_name,
        dB=dB
    ) for num in range(NUM_OF_TRAIN_WAV_FILES) for noise_name in NOISE_TYPE_TRAIN for dB in DBS_TRAIN]

    for clean_wav_path, noisy_wav_path in zip(paths_of_all_clean_wav, paths_of_all_noisy_wav):
        fname, _ = get_name_and_ext(clean_wav_path)
        if fname in target_fnames:
            shutil.copy(clean_wav_path, clean_wav_path.replace("test", "train"))
            shutil.copy(noisy_wav_path, noisy_wav_path.replace("test", "train"))


if __name__ == "__main__":
    release_data()
