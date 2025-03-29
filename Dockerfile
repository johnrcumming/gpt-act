FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

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
#RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN pip install jupyterlab transformers datasets tensorboard accelerate deepspeed fairscale 
RUN pip install wandb "huggingface_hub[cli]"

RUN git config --global user.name "gpt2act bot"
RUN git config --global user.email "noreply@undertheradar.ai"

# Add a build argument for CACHE_BUST
ARG CACHE_BUST=1
RUN git clone https://github.com/johnrcumming/gpt-act.git

# Add a build argument for WANDB_KEY and HUGGINGFACE_TOKEN
ARG WANDB_KEY
ARG HUGGINGFACE_TOKEN

# Set the environment variables
ENV WANDB_KEY=$WANDB_KEY
ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

RUN wandb login $WANDB_KEY
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

RUN chmod +x /root/gpt-act/launch.sh

# set wandb variables
ENV WANDB_LOG_MODEL="end"

EXPOSE 8888
EXPOSE 6006

ENTRYPOINT ["/root/gpt-act/launch.sh"]
