FROM apache/airflow:2.7.3
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

USER root
RUN sudo apt-get update && sudo apt-get install unzip
# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install

USER airflow