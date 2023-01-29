FROM python:3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN pip3 install torch torchvision torchaudio

WORKDIR /app

ADD ./apps .

EXPOSE 8005

RUN adduser --system --group --no-create-home appuser

RUN chown appuser:appuser -R /app

USER appuser

CMD ["python3","/app/app.py"]