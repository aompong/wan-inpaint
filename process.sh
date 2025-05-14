#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_dir> [target_fps] [width] [height] [crop|centercrop] [max_frames]"
    echo "Example:"
    echo "  $0 /dir/path 0 \"\" \"\" \"\" 41     # only trim number of frames"
    echo "  $0 /dir/path 8 480 480               # resize, pad to maintain aspect ratio"
    echo "  $0 /dir/path 25 720 1280 crop        # resize, crop to maintain aspect ratio"
    echo "  $0 /dir/path 16 1000 1000 centercrop # center crop, exact dimensions"
    exit 1
fi

input_dir="$1"
target_fps="${2:-0}" # 0 means no fps change
width="${3:-}"   
height="${4:-}"  
crop_mode="${5:-}"
max_frames="${6:-}"

if [ ! -d "$input_dir" ]; then
    echo "Error: Directory '$input_dir' does not exist."
    exit 1
fi


output_dir="$input_dir/process"
modifiers=()
[ "$target_fps" != "0" ] && modifiers+=("${target_fps}fps")
[ -n "$width" ] && [ -n "$height" ] && modifiers+=("${width}x${height}")
[ -n "$crop_mode" ] && modifiers+=("${crop_mode}")
[ -n "$max_frames" ] && modifiers+=("${max_frames}")
if [ ${#modifiers[@]} -gt 0 ]; then
    output_dir+="/$(IFS=_; echo "${modifiers[*]}")"
fi
mkdir -p "$output_dir"

for video in "$input_dir"/*.{mp4,mov,avi,mkv}; do
    if [ ! -f "$video" ]; then
        continue  
    fi

    base_name=$(basename -- "$video")
    output="$output_dir/${base_name%.*}.mp4"
    cmd="ffmpeg -i '$video'"

    [ "$target_fps" != "0" ] && cmd+=" -r $target_fps"

    if [ -n "$width" ] && [ -n "$height" ]; then
        if [ "$crop_mode" = "crop" ]; then
            cmd+=" -vf 'scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height}'"
        elif [ "$crop_mode" = "centercrop" ]; then
            cmd+=" -vf 'crop=${width}:${height}:(in_w-${width})/2:(in_h-${height})/2'"
        else
            cmd+=" -vf 'scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2'"
        fi
    fi

    [ -n "$max_frames" ] && cmd+=" -frames:v $max_frames"

    cmd+=" -c:v libx264 -crf 18 -preset slow -c:a copy '$output'"
    echo "Processing: $video → $output"
    eval "$cmd"
done

echo "Processing complete. Modified videos saved to input directory."
