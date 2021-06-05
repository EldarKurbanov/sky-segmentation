# syntax=docker/dockerfile:1
FROM ubuntu
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV TZ=Europe/Volgograd
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get -y install python3 python3-pip ffmpeg libsm6 libxext6 git
COPY requirements.txt requirements.txt
RUN pip install -q git+https://github.com/tensorflow/examples.git
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]