# Build Speech Enhancement Dataset

Build speech enhancement dataset using TIMIT and NoiseX92 corpus.

## Dependency package

- tqdm
- librosa

## Usage

Clone Repository:

```shell
git clone https://github.com/haoxiangsnr/Build-SE-Dataset.git
cd Build-SE-Dataset
```

Download TIMIT Corpus from https://github.com/philipperemy/timit

Extract it and put it in the `./data/TIMIT directory`:

```shell
data
├── NoiseX92
│   ├── babble.wav
│   ├── buccaneercockpit1.wav
│   ├── buccaneercockpit2.wav
│   ├── destroyerengine.wav
│   ├── destroyerops.wav
│   ├── f16.wav
│   ├── factoryfloor1.wav
│   ├── factoryfloor2.wav
│   ├── hfchannel.wav
│   ├── leopard.wav
│   ├── m109.wav
│   ├── machinegun.wav
│   ├── pinknoise.wav
│   ├── volvo.wav
│   └── whitenoise.wav
└── TIMIT
    ├── TIMIT
```

Configuring `./config.json`:

```json
{
    "release_dir": "release_timit",
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


Build dataset:

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