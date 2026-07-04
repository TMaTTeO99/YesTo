FROM python:3.12-slim
WORKDIR /usr/local/app

RUN apt-get update && apt-get install -y libportaudio2 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . ./
ENV DEBUG=true
CMD ["python3", "main.py"]
