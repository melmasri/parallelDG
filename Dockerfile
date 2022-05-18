FROM onceltuca/pygraphviz

RUN pip install pyrsistent==0.16.0
RUN pip install tabulate
RUN pip install parallelDG
#RUN pip install -e /home/mo/src/parallelDG/
#RUN python2 -m pip install parallelDG==0.3
