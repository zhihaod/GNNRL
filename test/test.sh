#!/bin/bash  


x=1
while [ $x -le 20 ]
do
  python main.py >> output.txt
done