import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# configurable
SRC_ROOT = Path("data/webcam_dataset")           # your raw webcam captures
DST_ROOT = Path("data/processed/webcam_clean")  # <--- Option C
IMG_SIZE = 64
PAD = 40  # pixels padding around hand bbox

mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation


def ensure_dirs():
    DST_ROOT.mkdir(parents=True, exist_ok=True)


def _apply_segmentation_mask(img, mask):
    
    mask = (mask > 0.5).astype(np.uint8)
    white = np.ones_like(img, dtype=np.uint8) * 255
    out = img * mask[:, :, None] + white * (1 - mask[:, :, None])
    return out.astype(np.uint8)


def preprocess_one_image(img_path, hands, seg):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    xmin = int(min(xs) * w) - PAD
    xmax = int(max(xs) * w) + PAD
    ymin = int(min(ys) * h) - PAD
    ymax = int(max(ys) * h) + PAD

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    if xmin >= xmax or ymin >= ymax:
        return None

    crop = img[ymin:ymax, xmin:xmax]

    # segmentation mask on full image -> crop region
    seg_res = seg.process(rgb)
    if seg_res and seg_res.segmentation_mask is not None:
        mask = seg_res.segmentation_mask
        mask_crop = mask[ymin:ymax, xmin:xmax]
        clean = _apply_segmentation_mask(crop, mask_crop)
    else:
        clean = crop

    # final resize
    clean = cv2.resize(clean, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return clean


def main(src_root: Path = SRC_ROOT, dst_root: Path = DST_ROOT):
    if not src_root.exists():
        print(f"Source folder does not exist: {src_root}")
        sys.exit(1)

    ensure_dirs()

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)
    seg = mp_selfie.SelfieSegmentation(model_selection=1)

    class_folders = [p for p in src_root.iterdir() if p.is_dir()]
    if len(class_folders) == 0:
        print("No class folders found in source root")
        sys.exit(1)

    for cls_folder in class_folders:
        cls_name = cls_folder.name
        out_folder = dst_root / cls_name
        out_folder.mkdir(parents=True, exist_ok=True)

        img_paths = [p for p in cls_folder.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if len(img_paths) == 0:
            print(f"Skipping empty class: {cls_name}")
            continue

        pbar = tqdm(img_paths, desc=f"{cls_name}", unit="img")
        saved = 0
        for p in pbar:
            try:
                proc = preprocess_one_image(p, hands, seg)
                if proc is None:
                    continue
                out_name = out_folder / p.name
                cv2.imwrite(str(out_name), proc)
                saved += 1
            except Exception:
                continue
        pbar.close()
        print(f"Saved {saved}/{len(img_paths)} images for class {cls_name}")

    hands.close()
    print("Preprocessing finished. Clean dataset at:", dst_root)


if __name__ == '__main__':
    main()
