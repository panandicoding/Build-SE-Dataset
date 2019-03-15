import glob
import json
import os
import random
import shutil
from pathlib import Path
import numpy as np

import librosa
from tqdm import tqdm

from utils import get_name_and_ext, add_noise_for_waveform, prepare_empty_dirs


def load_wavs(file_paths, limit=None, sr=16000, minimum_sampling=0):
    """根据文件路径的 list 逐个加载 wav 文件

    可以指定：
    - wav 文件需要满足的最小采样点数
    - 需要加载的 wav 文件数量，直到遍历完整个 list 或 满足了 limit 指定的数量要求

    Args:
        file_paths: 候选集合，其中采样点数大于 minimum_sampling 的 wav 才能被加载成功
        limit: 要求加载的数量上限
        sr: 采样率
        minimum_sampling: 最小采样点数

    """

    wavs = []
    i = 0

    if not limit:
        limit = len(file_paths)

    print("Loading wavs ...")
    while i < limit:
        wav, _ = librosa.load(file_paths[i], sr=sr)
        if len(wav) >= minimum_sampling:
            wavs.append(wav)
            i += 1

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
    for noise_path in tqdm(noise_wav_paths, desc="Loading noises: "):
        name, _ = get_name_and_ext(noise_path)
        wav, _ = librosa.load(noise_path, sr=16000)
        out[name] = wav

    return out

def add_noise_for_wavs(noise_ys_dict:dict, clean_ys, dbs, output_dir):
    """批量合成带噪语音

    将 noise_paths 中的噪声文件，按照各种 dbs，分别叠加在 clean_wav_paths 上
    最终的结果保存至 output_dir/noisy, output_dir/clean

    Args:
        noise_ys_dict(list): 噪声文件路径
        clean_ys(list): 纯净语音文件路径
        dbs(list): 信噪比
        output_dir(Path): 输出的目录，目录必须存在

    Returns:
        store: {
            "00001_babble_-1": {
                "clean": clean_y,
                "noisy": noisy_y
            },
            ...
        }
    """
    assert (output_dir / "clean").exists()
    assert (output_dir / "noisy").exists()

    store = {}

    for i, clean_y in tqdm(enumerate(clean_ys, 1), desc="Add noise for clean waveform"):
        for noise_type in noise_ys_dict.keys():
            for db in dbs:
                output_wav_basename_text = f"{str(i).zfill(4)}_{noise_type}_{db}"
                output_noisy_y_path = os.path.join(output_dir.as_posix(), "noisy", output_wav_basename_text + ".wav")
                output_clean_y_path = os.path.join(output_dir.as_posix(), "clean", output_wav_basename_text + ".wav")

                noisy_y = add_noise_for_waveform(clean_y, noise_ys_dict[noise_type][:len(clean_y)], int(db))
                assert len(noisy_y) == len(clean_y)

                librosa.output.write_wav(output_clean_y_path, clean_y, 16000)
                librosa.output.write_wav(output_noisy_y_path, noisy_y, 16000)

                store[output_wav_basename_text] = {
                    "noisy": noisy_y,
                    "clean": clean_y
                }

    return store


def main(config):
    data_dir = Path("data")
    timit_data_dir = data_dir / "TIMIT"  # 存放着 TIMIT 语料库
    noisex92_data_dir = data_dir / "NoiseX92"  # 存放着 NoiseX-92 噪声语料库
    release_dir = Path(config["release_dir"])
    release_dir_for_test = release_dir / "test"
    release_dir_for_train = release_dir / "train"

    prepare_empty_dirs([
        release_dir_for_train / "noisy",
        release_dir_for_train / "clean",
        release_dir_for_test / "noisy",
        release_dir_for_test / "clean",
    ])

    # Classification of TIMIT for train and test
    timit_wav_paths = librosa.util.find_files(timit_data_dir.as_posix(), ext=["WAV"], recurse=True)
    random.shuffle(timit_wav_paths)  # select wav file randomly
    clean_ys = load_wavs(
        timit_wav_paths,
        limit=config["test"]["num_of_utterance"],
        minimum_sampling=config["minimum_sampling"]
    )

    noisex92_wav_paths_list = [p for p in glob.glob(noisex92_data_dir.as_posix() + "/*.wav")]
    test_noise_paths = [p for p in noisex92_wav_paths_list if get_name_and_ext(p)[0] in config["test"]["noise_types"]]
    noise_ys_dict = load_noises(test_noise_paths)

    test_store = add_noise_for_wavs(  # Build test dataset
        noise_ys_dict=noise_ys_dict,
        clean_ys=clean_ys,
        dbs=config["test"]["dbs"],
        output_dir=release_dir_for_test
    )

    paths_of_test_clean_wav = sorted(glob.glob((release_dir_for_test / "clean").as_posix() + "/*.wav"))
    paths_of_test_noisy_wav = sorted(glob.glob((release_dir_for_test / "noisy").as_posix() + "/*.wav"))

    print("Build test dataset finish.")
    print("Select train dataset from test dataset...")

    # select train from test database
    train_basename_texts = []
    for i in range(config["train"]["num_of_utterance"]):
        for noisy_type in config["train"]["noise_types"]:
            for db in config["train"]["dbs"]:
                train_basename_texts.append(f"{str(i + 1).zfill(4)}_{noisy_type}_{db}")

    train_store = {}
    for clean_wav_path, noisy_wav_path in zip(paths_of_test_clean_wav, paths_of_test_noisy_wav):
        basename_text, _ = get_name_and_ext(clean_wav_path)
        if basename_text in train_basename_texts:
            shutil.copy(clean_wav_path, clean_wav_path.replace("test", "train"))
            shutil.copy(noisy_wav_path, noisy_wav_path.replace("test", "train"))

            train_store[basename_text] = test_store[basename_text]

    print("Select train dataset finshed. Begin saving numpy object file...")
    np.save((release_dir / "train.npy").as_posix(), train_store)
    np.save((release_dir / "test.npy").as_posix(), test_store)
    print(f"Build SE dataset finished, result In {release_dir}.")
    print("You can use command line to transfer release data to remote dir: ")
    print('\t time tar -c <local_release_dir> | pv | lz4 -B4 | ssh user@ip "lz4 -d | tar -xC <remote_dir>"')


if __name__ == "__main__":
    config = json.load(open("./config.json"))
    main(config)
