FROM python:3.10-slim

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY package_folder package_folder
COPY model model
COPY data/outputs data/outputs
COPY data/data_preprocessed data/data_preprocessed
COPY config config

CMD uvicorn package_folder.api_file:app --host 0.0.0.0 --port $PORT
