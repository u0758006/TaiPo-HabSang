# KhehUe-HapSing
自 https://github.com/fatchord/WaveRNN 來訓練。

## 安
- [dobi](https://github.com/dnephin/dobi)
- [docker](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
- [docker-compose](https://docs.docker.com/compose/install/)
- 設定docker權限`sudo usermod -aG docker $USER`

## 步
1. 先用`time dobi hazoi-ngiliau`，會下載客語教典詞音檔，下載好會恁樣：
```
1-ciidien-20190516/
├── corpus
│   ├── s_sound
│   │   ├── 00001.mp3
│   │   ├── 00002.mp3
│   │   ├── ...
│   │   └── 15486.mp3
│   ├── s_sound2
│   │   └── ...
│   ├── s_sound3
│   │   └── ...
│   ├── s_sound4
│   │   └── ...
│   ├── s_sound5
│   │   └── ...
│   └── s_sound6
│        └── ...
└── moe-hakkadict-main
    ├── README.md
    ├── 調值資料_raw
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).ods
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).pdf
    ├── 調值資料_uni
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    ├── 調型資料
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    └── 轉做調型資料.py
```
2. `time dobi zon-bienma`，毋愛頭尾無聲个部份，tacotron較會收斂，而且wave downsample 乜降做 16bits wav，盡尾合成較遽。
```
2-ciidien-20190516-16k/
├── corpus
│   ├── s_sound
│   │   ├── 00001.mp3.wav
│   │   ├── 00002.mp3.wav
│   │   ├── ...
│   │   └── 15486.mp3.wav
│   ├── s_sound2
│   │   └── ...
│   ├── s_sound3
│   │   └── ...
│   ├── s_sound4
│   │   └── ...
│   ├── s_sound5
│   │   └── ...
│   └── s_sound6
│        └── ...
└── moe-hakkadict-main
    ├── README.md
    ├── 調值資料_raw
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).ods
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).pdf
    ├── 調值資料_uni
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    ├── 調型資料
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    └── 轉做調型資料.py
```
3. `time dobi preprocess-tacotron`，準備tactorn格式，產生音檔長短`dataset.pkl`，音檔羅馬字對應`text_dict.pkl`、音檔頻譜`mel/`、音檔波形sample`quant/`。
```
3-ciidien-20190516-16k-MeuLid-data/
├── dataset.pkl
├── gta
├── mel
│   ├── 00001.mp3.npy
│   ├── 00002.mp3.npy
│   ├── ...
│   └── 15450.mp3.npy
├── quant
│   ├── 00001.mp3.npy
│   ├── 00002.mp3.npy
│   ├── ...
│   └── 15450.mp3.npy
└── text_dict.pkl
```
4. `time dobi tacotron`，訓練Tacotron模型。假使在tactorn訓練時節，愛產生`gta/`檔案，走`dobi tacotron-gta`。
5. `time dobi preprocess-wavernn`，因為太長个音檔無法度用訓練tacotron，會無法度coverage。`hparams.py`有設定`tts_max_mel_len`，故所`gta/`無全部音檔有。這指令照`gta/`檔案，產生wavernn愛个`dataset_wavernn.pkl`。
6. `time dobi wavernn`，訓練WaveRNN模型。
7. `time dobi huatsiann`，合成語句。

#### Pau--khi-lai
```
time dobi hokbu-khuanking
# GPU
docker run --rm -ti -e CUDA_VISIBLE_DEVICES=1 -v `pwd`/kiatko:/kiatko -p 5000:5000 i3thuan5/suisiann-wavernn:SuiSiann-WaveRNN-HokBu-fafoy
# CPU
docker run --rm -ti -e FORCE_CPU=True -v `pwd`/kiatko:/kiatko -p 5000:5000 i3thuan5/suisiann-wavernn:SuiSiann-WaveRNN-HokBu-fafoy
```

##### Tshi(舊)
Python
```python
from http.client import HTTPConnection
from urllib.parse import urlencode

taiBun='tak10-ke7 tsə2-hue1 lai7 tsʰit8-tʰə5 !'
參數 = urlencode({
    'taibun': taiBun,
    'sootsai': 'taiBun/tshi.wav',
})
headers = {
    "Content-type": "application/x-www-form-urlencoded",
    "Accept": "text/plain"
}
it_conn = HTTPConnection('hapsing', port=5000)
it_conn.request("POST", '/', 參數, headers)
it_conn.getresponse().read()
```
