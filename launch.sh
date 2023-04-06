#!/bin/sh

tensorboard --bind_all --port=6006 --logdir gpt-act/runs &
jupyter lab --ip=0.0.0.0 --allow-root --port=8888 