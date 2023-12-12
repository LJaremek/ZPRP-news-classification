#!/usr/bin/env just --justfile

run-server:
    python endpoints.py

create-env:
    conda create --name zprp python=3.10.9

activate-env:
    conda activate zprp

deactivate-env:
    conda deactivate

install-deps:
    conda install --file requirements.txt

dump-deps:
    pip list --format=freeze > requirements.txt

run-test:
    python -m unittest
