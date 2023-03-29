FROM python:3.9.8

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install --upgrade pip
RUN pip install -U pip wheel cmake
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  libgl1 -y
RUN pip install -r requirements1.txt

EXPOSE 8080

CMD python app.py