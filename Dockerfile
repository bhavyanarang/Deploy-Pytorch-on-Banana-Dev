FROM python:3.10.4

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY scripts /app/scripts
COPY models /app/models
COPY images /app/images

ADD server.py .
ADD app.py .

EXPOSE 8000
CMD python -u server.py