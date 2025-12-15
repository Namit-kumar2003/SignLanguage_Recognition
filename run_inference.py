import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque
import time

from src.model.cnn_lstm import CNNLSTM
from src.dataset.augmentations import get_val_aug
from src.utils.device import get_device

import mediapipe as mp
mp_hands = mp.solutions.hands



def pick_checkpoint(checkpoint_dir=Path('checkpoints')):
    best = checkpoint_dir / 'best_model.pth'
    if best.exists():
        return best
    
    candidates = sorted(checkpoint_dir.glob('epoch_*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 0:
        return candidates[0]
    raise FileNotFoundError('No checkpoint found in checkpoints/')


class ASLPredictor:
    def __init__(self, checkpoint_path, idx_to_class, device, img_size=64, num_frames=3):
        self.device = device
        self.idx_to_class = idx_to_class
        self.num_classes = len(idx_to_class)
        self.img_size = img_size
        self.num_frames = num_frames

        self.frame_buffer = deque(maxlen=num_frames)
        self.smooth_buffer = deque(maxlen=6)

        
        self.model = CNNLSTM(num_classes=self.num_classes, num_frames=num_frames)

        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        saved_sd = ckpt.get('model_state', ckpt)
        model_sd = self.model.state_dict()
        new_sd = {}
        for k, v in saved_sd.items():
            if k in model_sd and v.size() == model_sd[k].size():
                new_sd[k] = v
            else:
                print(f"SKIP param: {k} (shape mismatch or missing)")

        model_sd.update(new_sd)
        self.model.load_state_dict(model_sd)
        self.model.to(device)
        self.model.eval()

        self.transform = get_val_aug(img_size)

        
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def mediapipe_crop(self, frame, pad=48):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]
        xmin = int(min(xs) * w) - pad
        xmax = int(max(xs) * w) + pad
        ymin = int(min(ys) * h) - pad
        ymax = int(max(ys) * h) + pad
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(w, xmax); ymax = min(h, ymax)
        if xmin >= xmax or ymin >= ymax:
            return None
        crop = frame[ymin:ymax, xmin:xmax]
        return crop

    def smooth_vote(self):
        counts = {}
        for idx, p in self.smooth_buffer:
            counts[idx] = counts.get(idx, 0) + p
        best = max(counts, key=counts.get)
        return best, counts[best] / max(1, len(self.smooth_buffer))

    def predict(self, frame):
        crop = self.mediapipe_crop(frame)
        if crop is None:
            self.frame_buffer.clear()
            return None, None, None

        
        try:
            out = self.transform(image=crop)
            img = out['image']  # tensor C,H,W
        except Exception:
            # fallback to cv2 resize -> tensor
            img = cv2.resize(crop, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)

        self.frame_buffer.append(img)
        if len(self.frame_buffer) < self.num_frames:
            return None, None, crop

        seq = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0).to(self.device)  # (1,T,C,H,W)
        with torch.no_grad():
            logits = self.model(seq)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        prob = float(probs[idx])

        self.smooth_buffer.append((idx, prob))
        final_idx, final_prob = self.smooth_vote()

        return self.idx_to_class[final_idx], final_prob, crop


def build_idx_to_class(roots):
    classes = set()
    for r in roots:
        p = Path(r)
        if not p.exists():
            continue
        for d in p.iterdir():
            if d.is_dir():
                classes.add(d.name)
    classes = sorted(list(classes))
    return classes


def main():
    device = get_device()
    print('[DEVICE]', device)

    
    roots = [Path('data/processed/asl_alphabet_train'), Path('data/processed/webcam_clean')]
    idx_to_class = build_idx_to_class(roots)
    print('[CLASSES]', len(idx_to_class))

    ckpt_path = pick_checkpoint()
    print('[CKPT]', ckpt_path)

    predictor = ASLPredictor(checkpoint_path=ckpt_path, idx_to_class=idx_to_class, device=device)

    cap = cv2.VideoCapture(0)
    last_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred, prob, crop = predictor.predict(frame)

        
        if pred is not None:
            text = f"Pred: {pred} ({prob:.2f})"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

        
        if crop is not None:
            c = cv2.resize(crop, (200, 200))
            h, w = frame.shape[:2]
            x0 = w - 220
            y0 = 20
            frame[y0:y0+200, x0:x0+200] = c

        
        now = time.time()
        fps = 1.0 / (now - last_time + 1e-6)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('ASL Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
