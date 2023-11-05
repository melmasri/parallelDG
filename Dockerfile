
FROM python:2.7

RUN apt update -y
RUN apt install -y graphviz libgraphviz-dev
RUN apt install time 

RUN mkdir git
WORKDIR /git
RUN git clone https://github.com/melmasri/parallelDG.git
WORKDIR /git/parallelDG
RUN git fetch --all --tag # This is not triggered if the version is changed. It should be.
RUN git checkout v0.9.6 -b latest
RUN pip install -r requirements.txt
ENV PYTHONPATH /git:/git/parallelDG:/git/parallelDG/bin
ENV PATH /git/parallelDG/bin:$PATH
RUN chmod 755 bin/*





