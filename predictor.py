
from pathlib import Path
import torch
import torch.nn.functional as F

from src.model.cnn_lstm import CNNLSTM


class ASLPredictor:
    def __init__(self, checkpoint_path: str, num_classes: int, device: torch.device, num_frames: int = 3):
        self.device = device
        self.num_frames = num_frames
        self.num_classes = num_classes

        # instantiate model and load state
        self.model = CNNLSTM(num_classes=self.num_classes, feature_dim=256, num_frames=self.num_frames).to(self.device)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=self.device)

        # checkpoint may be a dict with 'model_state'
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            
            state = ckpt

        
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def predict(self, seq_tensor: torch.Tensor):
        
        with torch.no_grad():
            seq_tensor = seq_tensor.to(self.device)
            logits = self.model(seq_tensor)  # (1, num_classes)
            probs = F.softmax(logits, dim=1)
            prob, idx = torch.max(probs, dim=1)
            return int(idx.item()), float(prob.item()), probs.squeeze(0)
