# Dockerfile for running pytorch + jupyter with rocm6.2.4
FROM rocm/dev-ubuntu-24.04:6.2.4

RUN apt update 
RUN apt install python3-dev python3-pip python3-venv -y
WORKDIR /root

RUN python3 -m venv .venv
RUN ./.venv/bin/pip3 install wheel setuptools
RUN ./.venv/bin/pip3 install jupyter
RUN ./.venv/bin/pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2
RUN ./.venv/bin/pip3 install numpy scipy matplotlib pandas

WORKDIR /root/project

CMD /root/.venv/bin/jupyter lab --allow-root --no-browser --ip=0.0.0.0