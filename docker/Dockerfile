#FROM ubuntu:latest
FROM python:3.8.1
COPY ./api/app.py /exp/
COPY ./svm_Gamma=0.01_C=0.5.joblib /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install -r /exp/requirements.txt
WORKDIR /exp
ENV FLASK_APP=app
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
EXPOSE 5000
