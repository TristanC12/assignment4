# COMP 3610 Assignment 4

## Project files
- assignment4.ipynb
- app.py
- test_app.py
- Dockerfile
- docker-compose.yml
- requirements.txt
- README.md
- .gitignore
- .dockerignore
- models/

## Python version
Use Python 3.11.

## Install

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

## Notebook workflow
1. Run `assignment4.ipynb`
2. Let the notebook download and process the data
3. Train the two regression models
4. Log the runs in MLflow
5. Register the best model
6. Save the best model into `models/`

## Run the API

uvicorn app:app --reload

## Run the tests

pytest -q


## Run MLflow UI

mlflow ui --backend-store-uri sqlite:///mlflow.db


## Docker

docker build -t taxi-tip-api .
docker run -p 8000:8000 taxi-tip-api

## Docker Compose

docker compose up --build

## Swagger docs

![alt text](image.png)


