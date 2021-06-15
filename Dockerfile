# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.2.2-gpu-py3
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV TZ=Europe/Volgograd
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get -y install python3 python3-pip ffmpeg libsm6 libxext6 git
#RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
#RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH python3 setup.py install 
#RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
COPY requirements.txt requirements.txt
RUN pip install -q git+https://github.com/tensorflow/examples.git
RUN pip install --use-feature=2020-resolver -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["python3", "app.py"]
#CMD ["flask", "run"]