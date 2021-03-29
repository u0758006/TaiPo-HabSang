import argparse
from csv import DictReader
import os
from multiprocessing import Pool, cpu_count
from os.path import splitext, basename
from pathlib import Path
import pickle
from typing import Union

from utils import hparams as hp
from utils.display import *
from utils.dsp import *
from utils.files import get_files
from utils.paths import Paths


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError(
            '%r must be an integer greater than 0' % num)
    return n


parser = argparse.ArgumentParser(
    description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument(
    '--path', '-p',
    help='directly point to dataset path (overrides hparams.wav_path'
)
parser.add_argument(
    '--extension', '-e', metavar='EXT', default='.wav',
    help='file extension to search for in dataset folder'
)
parser.add_argument(
    '--num_workers', '-w', metavar='N', type=valid_n_workers,
    default=cpu_count() - 1,
    help='The number of worker threads to use for preprocessing'
)
parser.add_argument(
    '--hp_file', metavar='FILE', default='hparams.py',
    help='The file to use for the hyperparameters'
)
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path

extension = args.extension
path = args.path


def khehue(path: Union[str, Path], wav_files):
    csv_files = get_files(path, extension='.csv')

    u_tihleh = set()
    for wav_sootsai in wav_files:
        mia = basename(wav_sootsai)
        if mia.endswith('.wav'):
            u_tihleh.add(mia[:-4])

    text_dict = {}
    gigian = os.environ['GIGIAN']
    for csv_file in sorted(csv_files):
        if gigian in csv_file.name:
            with open(csv_file, encoding='utf-8') as f:
                for tsua in DictReader(f):
                    mia = basename(tsua['客語音檔'])
                    if mia in u_tihleh:
                        hj = tsua['客家語']
                        lmj = tsua['客語標音']
                        if (
                            '（' not in hj and '(' not in hj and '【' not in hj
                            and '（' not in lmj and '(' not in lmj
                            and '【' not in lmj and '/' not in lmj
                        ):
                            text_dict[mia] = lmj.replace('\xa0','')

    return text_dict


def convert_file(path: Path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(
            y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def process_wav(path: Path):
    wav_id = path.stem
    m, x = convert_file(path)
    np.save(paths.mel / f'{wav_id}.npy', m, allow_pickle=False)
    np.save(paths.quant / f'{wav_id}.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]


wav_files = get_files(path, extension)
paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

if len(wav_files) == 0:

    print('Please point wav_path in hparams.py to your dataset,')
    print('or use the --path option.\n')

else:

    if not hp.ignore_tts:

        text_dict = khehue(path, wav_files)

        with open(paths.data / 'text_dict.pkl', 'wb') as f:
            pickle.dump(text_dict, f)

    n_workers = max(1, args.num_workers)

    simple_table([
        ('Sample Rate', hp.sample_rate),
        ('Bit Depth', hp.bits),
        ('Mu Law', hp.mu_law),
        ('Hop Length', hp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}')
    ])

    aie = []
    for tong in wav_files:
        if tong.stem in text_dict:
            aie.append(tong)

    pool = Pool(processes=n_workers)
    dataset = []

    for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav, aie), 1):
        dataset += [(item_id, length)]
        bar = progbar(i, len(aie))
        message = f'{bar} {i}/{len(aie)} '
        stream(message)

    with open(paths.data / 'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
