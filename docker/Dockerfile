#
# docker build -f Dockerfile-gpu -t pytorch-gpu .
#

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG python_version=3.6
ARG DEBIAN_FRONTEND=noninteractive

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV QT_QPA_PLATFORM=offscreen
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# install conda
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    echo "/usr/local/cuda/extras/CUPTI/lib64" > /etc/ld.so.conf.d/libcputi.conf && \
    chmod a+x /etc/profile.d/conda.sh

RUN apt-get update && \
    apt-get install -y git wget curl bzip2 unzip gnupg2 tree libglib2.0 && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ./Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    conda install conda python=${python_version} six wheel numpy pillow opencv && \
    conda clean -yt && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN conda create --name cmr python=2.7 && \
    /bin/bash -c ". activate cmr && conda install -c pytorch -y pytorch torchvision && conda clean -yt" && \
    echo "source activate cmr" >> ~/.bashrc

# jupyter script
COPY start_jupyter.sh /usr/local/bin

# init script
COPY init_cmr.sh /usr/local/bin

WORKDIR /projects/cmr/
CMD ["bash"]
