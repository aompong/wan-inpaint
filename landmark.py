import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageFilter
import scipy.ndimage
from tqdm import tqdm
import os

MAX = 5000000
EXPAND=15
BLUR=15

INPUT='input/man.mp4'
MODEL='assets/face_landmarker.task'
DIR=f'output/landmark'
os.makedirs(DIR, exist_ok=True)
FILENAME=f'{INPUT.split("/")[-1].split(".")[0]}'

OUTER_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),       
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),  
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),          
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291)    
])
INNER_LIPS = frozenset([
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
])
INNER_ORDER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,  # Top inner (left to right)
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95   # Bottom inner (right to left)
]

# modified from https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/mask_nodes.py
def expand_mask(mask, expand=5, tapered_corners=True, blur_radius=3):
    mask = mask.astype(np.float32) / 255.0
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    current_expand = expand
    output = mask.copy()
    
    for _ in range(abs(round(current_expand))):
        if current_expand < 0:
            output = scipy.ndimage.grey_erosion(output, footprint=kernel)
        else:
            output = scipy.ndimage.grey_dilation(output, footprint=kernel)
    output = (output * 255).astype(np.uint8)
    
    if blur_radius > 0:
        pil_image = Image.fromarray(output)
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
        output = np.array(pil_image)
    
    return output

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total=min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX)
x1, y1 = (0, 0)

video_out, mask_out, overlay_out = None, None, None
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video_out = cv2.VideoWriter(f'{DIR}/{FILENAME}_video.mp4', fourcc, fps, (W, H))
mask_out = cv2.VideoWriter(f'{DIR}/{FILENAME}_mask.mp4', fourcc, fps, (W, H), isColor=False)
overlay_out = cv2.VideoWriter(f'{DIR}/{FILENAME}_overlay.mp4', fourcc, fps, (W, H))

frame_count = 0
progress_bar = tqdm(total=total, desc="Frame")
while cap.isOpened() and frame_count < MAX:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, int(1000*frame_count/fps))

    if not result.face_landmarks:
        print('skip')
        progress_bar.update(1) 
        frame_count += 1
        continue

    lms = result.face_landmarks[0]
    mask = np.zeros((H, W), dtype=np.uint8)
    lip_points = []
    for idx in INNER_ORDER:
        if idx < len(lms):
            x = int(lms[idx].x * W)
            y = int(lms[idx].y * H)
            lip_points.append([x, y])

    if len(lip_points) >= 3:
        lip_points = np.array(lip_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [lip_points], color=1)
        mask_frame = expand_mask(mask * 255, expand=EXPAND, blur_radius=BLUR)
        
        video_frame = frame[y1:y1+H, x1:x1+W]
        mask_frame = mask_frame[y1:y1+H, x1:x1+W]
        if video_out: video_out.write(video_frame)
        if mask_out: mask_out.write(mask_frame)

        mask_norm = mask_frame.astype(np.float32) / 255.0
        overlay_frame = (video_frame * mask_norm[:, :, np.newaxis]).astype(np.uint8)
        if overlay_out: overlay_out.write(overlay_frame)

    frame_count += 1
    progress_bar.update(1)

cap.release()
if video_out: video_out.release()
if mask_out: mask_out.release()
if overlay_out: overlay_out.release()
progress_bar.close()
print(f'save video to {DIR}')
