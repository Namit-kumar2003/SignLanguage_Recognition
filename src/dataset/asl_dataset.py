from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ASLFeatureSequenceDataset(Dataset):
    def __init__(self, roots, transform=None, num_frames=3, oversample=50):
        # normalize roots -> list of paths
        if isinstance(roots, (str, Path)):
            roots = [Path(roots)]
        else:
            roots = [Path(r) for r in roots]

        # verify
        valid_roots = [r for r in roots if r.exists()]
        if len(valid_roots) == 0:
            raise FileNotFoundError(f"[Dataset] Missing dataset folder(s): {roots}")

        self.roots = valid_roots
        self.transform = transform
        self.num_frames = num_frames
        self.oversample = oversample

        
        classes = set()
        class_images = {}
        for r in self.roots:
            for d in r.iterdir():
                if not d.is_dir():
                    continue
                cls = d.name
                imgs = [p for p in d.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                if len(imgs) == 0:
                    continue
                classes.add(cls)
                class_images.setdefault(cls, []).extend(imgs)

        self.classes = sorted(list(classes))
        if len(self.classes) == 0:
            raise RuntimeError(f"[Dataset] No classes found in roots: {self.roots}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_images = class_images
        self.num_classes = len(self.classes)

        print(f"[DATASET] Loaded {self.num_classes} classes from {len(self.roots)} roots")
        for c in self.classes:
            print(f"  - {c}: {len(self.class_images[c])} images")

    def __len__(self):
        return len(self.classes) * self.oversample

    def __getitem__(self, index):
        cls = self.classes[index % len(self.classes)]
        label = self.class_to_idx[cls]

        
        imgs = self.class_images[cls]
        if len(imgs) < self.num_frames:
            chosen = [random.choice(imgs) for _ in range(self.num_frames)]
        else:
            chosen = random.sample(imgs, k=self.num_frames)

        frames = []
        for p in chosen:
            img = Image.open(p).convert('RGB')
            img = np.array(img)

            if self.transform is None:
                
                from torchvision.transforms import ToTensor, Resize
                t = Resize((64, 64))
                img = t(Image.fromarray(img))
                img = ToTensor()(img)
            else:
                
                try:
                    img_out = self.transform(image=img)
                    img = img_out['image']
                except Exception:
                    # maybe torchvision-style
                    from torchvision.transforms import ToTensor, Resize
                    t = Resize((64, 64))
                    img = t(Image.fromarray(img))
                    img = ToTensor()(img)

            frames.append(img)

        seq = torch.stack(frames, dim=0)  # (T, C, H, W)
        return seq, label
