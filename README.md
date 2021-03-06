# Build Speech Enhancement Dataset

Build speech enhancement dataset using TIMIT and NoiseX92 corpus. The current project has very limited functions, please feel free to pull

## Dependencies 

- tqdm
- librosa

## Usage

Clone repository, the repository contains most of the NoiseX-92 corpus and needs patience:

```shell
git clone https://github.com/haoxiangsnr/Build-SE-Dataset.git 
cd Build-SE-Dataset
```

Download TIMIT Corpus from https://github.com/philipperemy/timit.

Put it in the `./data/TIMIT` directory, extract it:

```shell
sudo apt install unzip
unzip data/TIMIT/TIMIT.zip -d data/TIMIT
```

The directory structure is as follows：

```shell
data
├── NoiseX92
│   ├── babble.wav
│   ├── buccaneercockpit1.wav
│   ├── buccaneercockpit2.wav
│   ├── destroyerengine.wav
│   ├── destroyerops.wav
│   ├── f16.wav
│   ├── factoryfloor1.wav
│   ├── factoryfloor2.wav
│   ├── hfchannel.wav
│   ├── leopard.wav
│   ├── m109.wav
│   ├── machinegun.wav
│   ├── pinknoise.wav
│   ├── volvo.wav
│   └── whitenoise.wav
└── TIMIT
    └── data
        └── lisa
            └── data
                └── timit
```

Configuring `./config.json`

- If the same `release_dir` is specified, `release_dir` will be cleared first and then generated again in `release_dir`
- The `dbs` and `noise_types` in the training set must be a subset of the test set `dbs`, `noise_types`
- Use `minimum_sampling` to specify the minimum number of samples, and the TIMIT corpus that meets the requirements will be used (Default sr = 16000).

```json
{
    "release_dir": "release_timit",
    "minimum_sampling": 16384,
    "train": {
        "num_of_utterance": 2,
        "dbs": [0, -5, -10, -20],
        "noise_types": ["babble", "destroyerengine", "destroyerops", "factoryfloor1"]
    },
    "test": {
        "num_of_utterance": 6,
        "dbs": [-20, -17, -15, -12, -10, -7, -5, 3, 0, 5, -3],
        "noise_types": ["babble", "destroyerengine", "destroyerops", "factoryfloor1", "factoryfloor2"]
    }
}
```


Build speech enhancement dataset:

```shell

python main.py

Loading noises: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:20<00:00,  4.06s/it]
Loading wavs: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00,  7.79it/s]
Add noise for clean waveform: 6it [00:00, 32.46it/s]
Build Test Dataset Finish.
Select train dataset from test dataset...
Select train dataset finshed. Begin saving numpy object file...
Build SE dataset finished, result In release_timit.
You can use command line to transfer release data to remote dir: 
         time tar -c <local_release_dir> | pv | lz4 -B4 | ssh user@ip "lz4 -d | tar -xC <remote_dir>"
```

The dataset is as follows:

```shell
release_timit/
├── test
│   ├── clean # 0001_factoryfloor1_-5.wav, ...
│   └── noisy # 0001_factoryfloor1_-5.wav, ...
├── test.npy # {"0001_factoryfloor1_-5": {"noisy": noisy_y, "clean": clean_y}, ...}
├── train
│   ├── clean # 0001_factoryfloor1_-5.wav, ...
│   └── noisy # 0001_factoryfloor1_-5.wav, ...
└── train.npy # {"0001_factoryfloor1_-5": {"noisy": noisy_y, "clean": clean_y}, ...}
```
