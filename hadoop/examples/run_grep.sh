#!/bin/sh

#! /bin/sh

pref=""
while [[ $# > 0 ]]
do
key="$1"
case "$key" in
  -dst)
   dst=$2
   shift
   ;;
  -cl)
   cl="clean"
   ;;
  -pref)
   pref=$2
   shift
   ;;
  -r)
   r="run"
   ;;

  *)
   #unknown opt
   ;;

esac
shift
done


dst=/user/userdata/analysis/$dst

if [[ !  -z $cl ]]; then
    echo "!rm "$dst
    hdfs dfs -rm -r $dst 
fi

if [[ ! -z $r ]]; then
    src=$(cat sessions_20140303_20140526.txt)
    echo "from: "$src
    echo "to dir: "$dst
    hadoop jar grep.jar GrepSessions $src $dst
fi

