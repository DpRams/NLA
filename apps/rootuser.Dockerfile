# Base image 是 python:3.7
FROM python:3.9

# requirements.txt 裡有我們需要的套件資訊， 
# 把他複製到container裡，路徑為 /app/requirements.txt
COPY ./requirements.txt /app/requirements.txt

# pip是python的套件管理工具
RUN pip install -r /app/requirements.txt
RUN pip3 install torch torchvision torchaudio

# 切換到container裡的 /app 路徑作為工作目錄 
WORKDIR /app

# 把本地端myapp資料夾複製到container的當前目錄 (/app)
ADD ./docker_apps .

# 8002 是我們服務所在的port
EXPOSE 8002

#在系統中加入一個新system user 和 group，名稱皆為appuser
RUN adduser --system --group --no-create-home appuser

#把/app 這個directory的擁有指定權指定到appuser
RUN chown appuser:appuser -R /app

#把container 的user轉到appuser
USER appuser

# CMD代表command，當你啟動這個container時，會預設執行這個指令
CMD ["python3","/app/app.py"]