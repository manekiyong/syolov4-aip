# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM nvcr.io/nvidia/pytorch:21.08-py3
FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install libgl1 -y
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
ADD /src /src
WORKDIR /src/mish-cuda/
RUN python setup.py build install

WORKDIR /src

CMD ["./train.sh"]
