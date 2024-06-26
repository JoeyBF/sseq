#/bin/bash
DATE=`date --iso-8601=seconds | sed "s/\(.*\)-.*/\1/" | sed "s/\:/-/g"`
old="$IFS"
IFS="-"
args_str=`echo $* | sed "s/_//g" | sed "s/ /-/g"`
IFS=$old
FILENAME="cachegrind.out.${DATE}_${args_str}"
nice -n +14 valgrind --tool=callgrind --callgrind-out-file=$FILENAME ./target/release/ext $*
