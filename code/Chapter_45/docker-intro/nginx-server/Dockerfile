# base image for building container
FROM docker.io/nginx
# add maintainer label
LABEL maintainer="dvdbisong@gmail.com"
# copy html file from local machine to container filesystem
COPY html/index.html /usr/share/nginx/html
# port to expose to the container
EXPOSE 80