FROM python:3.9

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-pip
RUN apt install pandoc

RUN pip install  -r requirements.txt

COPY . .