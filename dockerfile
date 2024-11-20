
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app


RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

