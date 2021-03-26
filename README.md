# KhehUe-HapSing
自 https://github.com/fatchord/WaveRNN 來訓練。

## 安
- [dobi](https://github.com/dnephin/dobi)
- [docker](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
- [docker-compose](https://docs.docker.com/compose/install/)
- 設定docker權限`sudo usermod -aG docker $USER`

## 步
1. 先用`time dobi liah-giliau`，會掠客語能力認證，掠好會生做按呢
```
Elearning-202102/
├── csv_imtong
│   ├── da3-1.csv
│   ├── da3-2.csv
│   ├── daw.csv
│   ├── ha3-1.csv
│   ├── ha3-2.csv
│   ├── haw.csv
│   ├── rh3-1.csv
│   ├── rh3-2.csv
│   ├── rhw.csv
│   ├── si3-1.csv
│   ├── si3-2.csv
│   ├── siw.csv
│   ├── zh3-1.csv
│   ├── zh3-2.csv
│   └── zhw.csv
└── mp3
    ├── 1da-01-001.mp3
    ├── 1da-01-001s.mp3
    ├── 1da-01-002.mp3
    ├── 1da-01-002s.mp3
    ├── ...
    ├── zh-18-157.mp3
    └── zh-18-157s.mp3
```
2. `time dobi giliau-pianma`，wave downsample 閣降做 16bits，上尾合成較緊
3. `time dobi preprocess-tacotron`，準備tactorn格式。
4. `time dobi tacotron`，訓練Tacotron模型。若是tī tactorn訓練中，欲產生gta檔案，走`dobi tacotron-gta`。
5. `time dobi preprocess-wavernn`，照gta檔案，產生wavernn需要ê`dataset.pkl`
6. `time dobi wavernn`，訓練WaveRNN模型
7. `time dobi huatsiann`，合成語句

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
