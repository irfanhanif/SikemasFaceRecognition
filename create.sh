#!/bin/bash

if [ "$EUID" -ne 0 ]
then
  echo "Failed. Please run as root"
  exit
elif [ -z "$1" ]
then
  echo "Failed. Please specify class name."
  exit
else
  cd kelas
  mkdir $1
  if (( $? == 0 )); then
    echo "Folder $1 created."
    echo "Creating sub-files.."
    cd $1
    mkdir training-images
    mkdir aligned-images
    mkdir generated-embeddings
    cp ../../train.sh .
    cp ../../classify.py .
    cp ../../preprocessing.py .
    if (( $? == 0 )); then
      echo "Sucessfully create sub-files"
    else
      cd ..
      rm -rf $1
      echo "Failed. Something caused errors."
      exit
    fi
  fi
fi
