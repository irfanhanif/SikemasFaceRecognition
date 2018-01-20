#!/bin/bash
# training something

if [ "$EUID" = 0 ]
then
  echo "Failed. Please do not run as root"
  exit
else
  ../../openface/util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
  ../../openface/batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
  python preprocessing.py
  python classify.py
fi
