FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

# for just prediction 
CMD ["python", "app.py"]