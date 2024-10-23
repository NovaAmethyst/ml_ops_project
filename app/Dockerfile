FROM python:3.11-bullseye

RUN pip install "fastapi[standard]"
COPY requirements.txt regression.joblib ./
RUN pip install -r requirements.txt
COPY backend.py ./app.py

CMD ["fastapi", "dev", "app.py", "--host", "0.0.0.0"]
