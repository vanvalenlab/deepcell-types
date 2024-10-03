FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN python -m pip install --upgrade pip

RUN apt-get update -y && apt-get install -y git

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

COPY . .

RUN python -m pip install deepcelltypes-kit/
