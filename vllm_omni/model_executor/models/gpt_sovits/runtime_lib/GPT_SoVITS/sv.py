import sys
import os
import torch

_PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
_ERES2NET_DIR = os.path.join(_PACKAGE_ROOT, "eres2net")
sv_path = os.path.join(_PACKAGE_ROOT, "pretrained_models", "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
if _ERES2NET_DIR not in sys.path:
    sys.path.insert(0, _ERES2NET_DIR)
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi


class SV:
    def __init__(self, device, is_half):
        pretrained_state = torch.load(sv_path, map_location="cpu", weights_only=False)
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        if is_half == False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav):
        with torch.no_grad():
            model_param = next(self.embedding_model.parameters())
            model_device = model_param.device
            model_dtype = model_param.dtype
            wav = wav.to(dtype=torch.float32)
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            feat = feat.to(device=model_device, dtype=model_dtype)
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb
