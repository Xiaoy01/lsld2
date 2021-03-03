#!/bin/bash
# Script to split a file [filename] containing a COO matrix into multiple
# files with sizes less than or equal to [max_size]. Assumes that each line of
# [filename] is a single entry in the COO matrix and max_size is in megabytes.

# Arguments
filename=$1
max_size=$2

# DEBUG
echo $filename
echo $max_size
pwd

# Verify that [filename] exists
if [ ! -f "$filename" ]; then
    echo "$filename does not exist."
    exit 1
fi

# Get size of [filename] in megabytes
let "file_size = $(wc -c $filename | cut -d ' ' -f2) / 1000000"

# Verify that [filename] needs splitting
if [ $file_size -lt $max_size ]; then
    echo "$filename does not need splitting."
    exit 1
fi

# Determine number of files to split [filename] into
let "num_files = 1 + $file_size / $max_size"

# Ensure that output directory name is not taken and create it
dirname="/"$($filename | cut -f 1 -d '.')"SPLIT"
if [ -d "$dirname"]; then
    echo "$dirname is already a directory."
    exit 1
else
    mkdir $dirname
fi

# Determine line numbers to split [filename] on and split and write
line_count=$(wc -l < $filename)
let "lines_per_file = $line_count / $num_files"
for start_line in $(seq 0 $lines_per_file $((lines_per_file * (num_files - 1)))); do
    let "end_line = $((start_line + lines_per_file))"
    let "end_line = $((end_line < line_count ? end_line : line_count))"
    sed -n "$start_line,$end_line p" $filename > $(dirname"/"$start_line".txt")
done
