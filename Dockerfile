FROM pytorch/pytorch
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3-pip
RUN pip install -r requirements.txt
