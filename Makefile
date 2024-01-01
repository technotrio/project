install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

build:
	docker build -t project .

run:
	docker run -p 8080:8080 project

lint:
	pylint --disable=C,R app.py