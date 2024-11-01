# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /patient_model_api

ADD /patient_model_api/requirements.txt .
ADD /patient_model_api/*.whl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

RUN rm *.whl

# copy application files
ADD /patient_model_api/app/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
