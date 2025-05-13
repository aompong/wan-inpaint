#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input.mp4>"
    exit 1
fi
input_mp4="$1"
if [ ! -f "$input_mp4" ]; then
    echo "Error: Directory '$input_mp4' does not exist."
    exit 1
fi

ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,nb_frames,duration,codec_name \
    -of default=noprint_wrappers=1 \
    "$input_mp4"


# ❯ chmod +x info.sh
# ❯ ./info.sh /path/to/your/video.mp4
# codec_name=h264
# width=2200
# height=3208
# r_frame_rate=25/1
# duration=22.040000
# nb_frames=551
