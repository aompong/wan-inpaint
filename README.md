# wan-inpaint

## Install
```bash
conda create -f env/environment.yml
conda activate wan-inpaint
```

## Run

### T2V
```bash
export HF_HUB_CACHE=/ist-nas/ist-share/vision/huggingface_hub/
python example/example_t2v_1.3b.py
```

## Create masked video
```bash
python landmark.py
```

## Scripts
- `info.sh` prints video summary
- `hstack.sh` concatenates videos horizontally
- `process.sh` changes fps, resizes, crops, trims frames
