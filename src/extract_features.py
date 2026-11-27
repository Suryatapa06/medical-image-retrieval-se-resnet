"""Extract deep feature vectors for an image folder and save them to CSV.

Folder layout: <root>/<class_name>/<image files>
CSV columns:   image_name, label, relative_path, feature_vector (space-separated)

Example:
    python src/extract_features.py --root Skin_Disease --out feats_stage4_r50_d64.csv \
        --stage 4 --backbone resnet50 --dim 64
"""
import argparse
import csv
import os

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import MultiStageFeatureResNet

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="image folder with one sub-folder per class")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "resnet101", "resnet152"])
    p.add_argument("--dim", type=int, default=64, help="reduced feature dimension per stage")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    # The SE blocks and 1x1 reduction convs are randomly initialised and NOT trained,
    # so fix the seed and build the model ONCE: every image must use the same weights.
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStageFeatureResNet(a.backbone, "IMAGENET1K_V1", a.dim, a.stage).to(device).eval()

    items = []
    for label in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, label)
        if os.path.isdir(d):
            items += [(label, f) for f in sorted(os.listdir(d)) if f.lower().endswith(EXTS)]

    with open(a.out, "w", newline="") as fh, torch.no_grad():
        w = csv.writer(fh)
        w.writerow(["image_name", "label", "relative_path", "feature_vector"])
        for label, name in tqdm(items, desc="Extracting features", unit="image"):
            path = os.path.join(a.root, label, name)
            img = TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            vec = model(img).squeeze(0).tolist()
            w.writerow([name, label, f"{os.path.basename(a.root)}/{label}/{name}",
                        " ".join(map(str, vec))])
    print(f"Saved {len(items)} feature vectors to {a.out}")


if __name__ == "__main__":
    main()
