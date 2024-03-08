FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

# 日本語環境
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG ja_JP.UTF-8

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   unzip \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace
RUN curl -L  "https://moji.or.jp/wp-content/ipafont/IPAexfont/IPAexfont00401.zip" > font.zip
RUN unzip font.zip
RUN cp /workspace/IPAexfont00401/ipaexg.ttf /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/ipaexg.ttf
RUN echo "font.family : IPAexGothic" >>  /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
RUN rm -r ./.cache

COPY . transformers/
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

CMD ["/bin/bash"]
