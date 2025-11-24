# !/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/drive/1gzQuAcZstMG28_R2vwAcqzmuXMZbQltf#scrollTo=CtrtBV54Ena3
  sleep 3600
done
