FROM ubuntu:latest
SHELL ["/bin/bash", "-c"]
USER root
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-full python3-pip python3-dev gcc g++ gfortran make pkg-config cmake \
    libsuitesparse-dev liblapack-dev libopenblas-dev libfftw3-dev mpich \
    libboost-all-dev &&  \
    rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

COPY ./ /home/pys4/

WORKDIR /home/pys4
RUN python3 -m venv .pys4 && . .pys4/bin/activate &&  \
    pip install ninja meson-python 'numpy>=1.20' \
    && pip install . --no-build-isolation

WORKDIR /home

RUN echo -e ". /home/pys4/.pys4/bin/activate" >>~/.bashrc
RUN . ~/.bashrc
