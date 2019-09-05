#
# docker build -f Dockerfile -t cmr .
#

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG python_version=3.6
ARG miniconda_file=Miniconda3-latest-Linux-x86_64.sh
ARG DEBIAN_FRONTEND=noninteractive

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV QT_QPA_PLATFORM=offscreen
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

USER root

# install conda
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    echo "/usr/local/cuda/extras/CUPTI/lib64" > /etc/ld.so.conf.d/libcputi.conf && \
    chmod a+x /etc/profile.d/conda.sh

RUN apt-get update && \
    apt-get install -y git wget curl bzip2 unzip gnupg2 tree libglib2.0 && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y octave && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    mkdir -p ~/.conda && \
    wget --quiet https://repo.continuum.io/miniconda/${miniconda_file} -O Miniconda3.sh && \
    /bin/bash ./Miniconda3.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3.sh && \
    conda install -y conda python=${python_version} six wheel numpy pillow opencv && \
    conda clean -yt

RUN conda install -c pytorch -y pytorch torchvision && \
    conda clean -yt

# jupyter script
COPY start_jupyter.sh /usr/local/bin

# init script
COPY init_cmr.sh /usr/local/bin

# test script
COPY test_gpu.py /usr/local/bin

# add user... purely needed for octave-gui to run from docker to X11
RUN useradd -ms /bin/bash cmr

EXPOSE 8888
EXPOSE 8097

WORKDIR /cmr/
CMD ["bash"]
