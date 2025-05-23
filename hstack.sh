#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi
dir="$1"
if [ ! -d "$dir" ]; then
    echo "Error: Directory '$dir' does not exist."
    exit 1
fi

mkdir -p "$dir/hstack"
output="$dir/hstack/combined.mp4"
videos=("$dir"/*.mp4) # get all mp4s in dir
if [ ${#videos[@]} -lt 2 ]; then
    echo "Error: At least 2 MP4 videos are required in the directory."
    exit 1
fi


inputs=()
filter=""
map_inputs=""
for ((i=0; i<${#videos[@]}; i++)); do
    inputs+=("-i" "${videos[$i]}")
    filename=$(basename "${videos[$i]}" .mp4)  # Extract filename without extension
    filter+="[${i}:v]drawtext=text='${filename}':fontsize=24:fontcolor=white:x=10:y=10[v${i}];"
    map_inputs+="[v${i}]"
done
filter+="${map_inputs}hstack=inputs=${#videos[@]}[v]"


ffmpeg "${inputs[@]}" -filter_complex "$filter" -map "[v]" "$output"
echo "Successfully stacked videos horizontally. Output: $output"

