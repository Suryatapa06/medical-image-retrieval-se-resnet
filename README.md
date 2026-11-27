# Attention Enhanced Multi-Stage ResNet for Medical Image Retrieval

Feature-extraction code for content-based medical image retrieval (CBIR) using a
ResNet backbone with Squeeze-and-Excitation (SE) channel attention in every residual
stage and progressive multi-stage feature fusion.

Related paper: *Attention Enhanced Visual Feature Extraction for Medical Image Retrieval
Using Multi-stage Residual Networks*, Proc. 2nd International Symposium on Artificial
Intelligence (ISAI 2025), Springer LNNS vol. 1536, pp. 141-156.
DOI: [10.1007/978-981-96-9239-2_11](https://link.springer.com/chapter/10.1007/978-981-96-9239-2_11)

## Method
1. Pretrained ResNet-50/101/152 stem and four residual stages (ImageNet weights).
2. An SE block recalibrates the channels after each stage.
3. Each stage output is reduced by a 1x1 convolution, pooled to 3x3 and added to the
   previous stages (progressive fusion).
4. Global average pooling gives one compact feature vector per image, used for
   similarity-based retrieval.

| Stage option | Features fused |
|---|---|
| `--stage 1` | stage 1 |
| `--stage 2` | stages 1 + 2 |
| `--stage 3` | stages 1 + 2 + 3 |
| `--stage 4` | stages 1 + 2 + 3 + 4 (full model) |

## Repository contents
- `src/model.py`: SE block and multi-stage feature extractor (stage 1-4 selectable).
- `src/extract_features.py`: extracts features for an image folder and writes a CSV.
- `notebooks/stage1-4_original.ipynb`: the original Colab notebooks for each stage
  (outputs cleared). Note: these rebuild the model for every image, while `src/` builds it once.

## Usage
```bash
pip install -r requirements.txt
# images arranged as <root>/<class_name>/<image files>
python src/extract_features.py --root path/to/images --out features.csv \
    --stage 4 --backbone resnet50 --dim 64
```

## Notes
- The SE blocks and 1x1 reduction convolutions are randomly initialised (fixed seed) and
  are not trained in this code. Only the ResNet backbone uses pretrained ImageNet weights.
- The notebooks in this repository were run on the ISIC skin lesion dataset
  (Kaggle: `shubhamdasin/skin-isic`). The paper evaluates retrieval on a curated chest
  X-ray dataset (COVID-19, Normal, Bacterial and Viral Pneumonia; 1,250 images per class).
- This repository contains the feature-extraction part of the pipeline. The retrieval
  evaluation (precision, recall, F-score at top-k) is not included.
- Datasets, feature CSVs and model weights are not included.

## My contribution
Group B.Tech final-year project at MCKV Institute of Engineering. *(Edit this line:
describe your own part, e.g. feature extraction code, experiments, paper writing.)*

## Authors
Group project under the guidance of Mr. Shukla Mondal, Department of Computer Science
and Engineering (Data Science), MCKV Institute of Engineering.
