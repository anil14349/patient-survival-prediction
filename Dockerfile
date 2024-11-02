# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /patient_model_api

ADD /patient_model_api/requirements.txt .
#ADD /patient_model/trained_models/xgboost-model.pkl ./patient_model/trained_models/xgboost-model.pkl

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

#RUN rm *.whl

# copy application files
ADD /patient_model_api/app/* ./app/
ADD /patient_model/trained_models/* ./patient_model/trained_models/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
