FROM python:3.10.6
WORKDIR /mlflow
COPY requirements.txt /mlflow/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /mlflow
EXPOSE 8080
CMD ["python", "app.py"]
