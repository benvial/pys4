FROM ubuntu:latest
USER root
SHELL ["/bin/bash", "-c"]
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-full python3-pip python3-dev gcc g++ pkg-config cmake \
    libsuitesparse-dev liblapack-dev libopenblas-dev libfftw3-dev mpich \
    libboost-all-dev imagemagick povray &&  \
    rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

COPY ./ /home/pys4/

WORKDIR /home/pys4

RUN python3 -m venv .pys4 && . .pys4/bin/activate &&  pip install . -v

RUN echo -e ". /home/pys4/.pys4/bin/activate" >>~/.bashrc
RUN . ~/.bashrc
RUN pip install pytest pytest-cov && pytest

WORKDIR /home
