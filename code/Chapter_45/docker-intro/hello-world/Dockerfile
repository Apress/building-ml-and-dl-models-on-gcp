# base image for building container
FROM docker.io/alpine
# add maintainer label
LABEL maintainer="dvdbisong@gmail.com"
# copy script from local machine to container filesystem
COPY date-script.sh /date-script.sh
# execute script
CMD sh date-script.sh