FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

WORKDIR /root

RUN apt-get update

RUN apt-get install -y wget bash openssh-server openssh-client git nvtop

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init
RUN conda --version

RUN conda install python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
RUN pip install jupyterlab transformers datasets tensorboard

RUN mkdir /root/.ssh
ADD id_rsa /root/.ssh/
RUN chmod -R 700 ~/.ssh/id_rsa

RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git config --global user.name "John Cumming"
RUN git config --global user.email "johnrcumming@gmail.com"

RUN git clone git@github.com:johnrcumming/gpt-act.git

RUN chmod +x /root/gpt-act/launch.sh

EXPOSE 8888
EXPOSE 6006
EXPOSE 22

ENTRYPOINT ["/root/gpt-act/launch.sh"]