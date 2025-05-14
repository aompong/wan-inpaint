# wan-inpaint

## Install
```bash
conda env create -f env/environment.yml
conda activate wan-inpaint
```

## Run

### SDEdit
```bash
export HF_HUB_CACHE=/ist-nas/ist-share/vision/huggingface_hub/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python sdedit.py
```

## Create masked video
```bash
python landmark.py
```

## Scripts
- `info.sh` prints video summary
- `hstack.sh` concatenates videos horizontally
- `process.sh` changes fps, resizes, crops, trims frames
