FROM bentoml/model-server:0.13.0

COPY . /bento
WORKDIR /bento

RUN pip install -r requirements.txt

ENV BENTOML_HOME=/bento
ENV BENTOML_API_NAME=MyService
ENV BENTOML_API_HANDLER=MyService
ENV BENTOML_ARCHIVE_PATH=/bento/MyService

EXPOSE 5000

CMD ["bentoml", "serve", "/bento/MyService"]
