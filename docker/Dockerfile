FROM tiangolo/uvicorn-gunicorn:python3.11

ARG DB_CONNECTION
ARG MODE

ENV SQLALCHEMY_DATABASE_URI=${DB_CONNECTION}
ENV MODE=${MODE}

COPY ./requirements/base.txt .
RUN pip install -r base.txt

COPY ./scripts /app
COPY ./image_segmentation /app
WORKDIR /app
