#!/usr/bin/env bash

while [ 1 ]; do
  ./gen_test_train_data.py
  killall Preview
  open test.png
  sleep 5
done
