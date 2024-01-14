#!/usr/bin/env just --justfile

run-server:
    python -m src.server.endpoints

create-env:
    conda create --name zprp python=3.10.9

activate-env:
    conda activate zprp

deactivate-env:
    conda deactivate

install-deps:
    pip install -r requirements.txt

run-test:
    python -m unittest

train:
    python3 -m src.models.train