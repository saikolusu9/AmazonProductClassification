FROM frolvlad/alpine-python-machinelearning:latest

WORKDIR /app

COPY . /app
EXPOSE 4000

ENTRYPOINT  ["python"]

CMD ["app.py"]
