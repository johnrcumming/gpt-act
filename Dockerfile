FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

WORKDIR /root

RUN apt-get update

RUN apt-get install -y wget bash openssh-server openssh-client git nvtop curl
RUN apt-get install -y apt-transport-https ca-certificates gnupg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y
            


RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init
RUN conda --version

RUN conda install python=3.10 
# RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install jupyterlab transformers datasets tensorboard accelerate deepspeed fairscale wandb

RUN mkdir /root/.ssh
ADD id_rsa /root/.ssh/
RUN chmod -R 700 ~/.ssh/id_rsa

RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git config --global user.name "John Cumming"
RUN git config --global user.email "johnrcumming@gmail.com"

RUN git clone -b moe_cascade git@github.com:johnrcumming/gpt-act.git

RUN chmod +x /root/gpt-act/train.sh

# set wandb variables
ENV WANDB_LOG_MODEL="end"

EXPOSE 8888
EXPOSE 6006
EXPOSE 22

ENTRYPOINT ["/root/gpt-act/train.sh"]
