# start by pulling the python image
FROM ubuntu:latest

# install build tools
RUN apt update
RUN apt install build-essential python3 pip -y

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader averaged_perceptron_tagger
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader omw-1.4

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python3" ]

CMD ["predict_flask.py" ]