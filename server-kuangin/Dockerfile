FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN apt-get update && \
  apt-get install -y git vim-tiny \
    libsndfile1 # librosa output

RUN git clone https://github.com/i3thuan5/WaveRNN.git /WaveRNN
WORKDIR /WaveRNN

ENV LANG C.UTF-8
RUN pip install -r requirements.txt

#RUN apt-get update && \
  #apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools

#RUN pip install --upgrade librosa
#RUN git pull # 20191125

RUN pip install flask gunicorn

COPY ./gen_tacotron.py zugi.py

COPY tshamsoo/hparams.py .
COPY tshamsoo/text_init.py utils/text/__init__.py
COPY tshamsoo/symbols.py utils/text/symbols.py
CMD gunicorn \
  -b 0.0.0.0:5000 \
  --timeout 60 \
  --log-level debug \
  zugi:app
