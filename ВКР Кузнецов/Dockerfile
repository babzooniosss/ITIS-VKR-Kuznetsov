FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY install.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN python install.py

COPY . .

RUN mkdir -p data output

ENV PYTHONUNBUFFERED=1

CMD ["python", "start.py"] 