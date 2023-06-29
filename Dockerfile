FROM python:3.10.11

WORKDIR /src

ADD ./ /src

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 

CMD ["python", "api.py"]