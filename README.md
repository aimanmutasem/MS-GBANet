# MS-GBANet

This repository hosts the initial public release of **MS-GBANet**, focused **on polyp segmentation** (training/testing) in colonoscopy images. The code is implemented in PyTorch and uses boundary-aware attention for accurate segmentation.

<p align="center"> <img src="https://github.com/aimanmutasem/MS-GBANet/blob/main/Msgbanet-architecture.png" alt="MS-GBANet architecture" width="720"> </p>

 🔜 **Coming soon:** implementations for additional datasets/modalities, ablation scripts, and pretrained weights.

 ## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/aimanmutasem/MS-GBANet.git
    cd MS-GBANe
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the training and test datasets from [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view), then extract/move them into the project directory at `./data/polyp/`.

4. Download the pretrained [PVTv2 model](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV), then move the .pth file to ./pretrained_pth/ for initialization.
   
## Usage

### Training
Train MS-GBANet on the polyp dataset (using GPU 0):
```sh
CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py
```

### Testing 
Evaluate on the polyp test set (using GPU 0):
```sh
CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py
```

## Directory Structure

- `lib/models_timm/`: Model and hub utilities
- `test_polyp.py`: Main script for testing and evaluation
- `data/polyp/`: Datasets directory 
- `model_pth/`: Pretrained model weights

## Citation

The paper is under review, : ) .

## License

Licensed under the Academic Non-Commercial License v1.0 (ANCL-1.0). See LICENSE for details.

---

For questions or issues, please open an [issue](https://github.com/aimanmutasem/MS-GBANet/issues).

