# MS-GBANet

This repository hosts the initial public release of **MS-GBANet**, focused **on polyp segmentation** (training/testing) in colonoscopy images. The code is implemented in PyTorch and uses boundary-aware attention for accurate segmentation.

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
5. 
## Usage

### Testing

To evaluate the model on the test datasets, run:

```sh
python test_polyp.py --model_path ./model_pth/Polyp_MSGBANET_img_size352bs4_Run1/Polyp_MSGBANET_img_size352bs4_Run1-best.pth
```

