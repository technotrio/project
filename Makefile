install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

build:
	docker build -t mflow:lts .

run:
	docker run -p 8080:8080 mlflow

lint:
	pylint --disable=C,R app.py