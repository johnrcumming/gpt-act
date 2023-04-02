FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget

RUN apt-get install git -y

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init
RUN conda --version

RUN conda install python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
RUN conda install jupyterlab

RUN mkdir /root/.ssh
ADD id_rsa /root/.ssh/
RUN chmod -R 700 ~/.ssh/id_rsa

RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git config --global user.name "John Cumming"
RUN git config --global user.email "johnrcummig@gmail.com"

RUN git clone git@github.com:johnrcumming/gpt-act.git

EXPOSE 8888

