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

You can adjust parameters such as `--img_size`, `--test_path`, and `--skip_aggregation` as needed.

### Model Card and Hugging Face Hub

To push your trained model to the Hugging Face Hub, use the [`push_to_hf_hub`](LLM_MSGCNET/msgbanet-main/lib/models_timm/hub.py) function. This will create a model card (`README.md`) with tags for image classification and timm.

## Directory Structure

- `lib/models_timm/`: Model and hub utilities
- `test_polyp.py`: Main script for testing and evaluation
- `data/polyp/TestDataset/`: Test datasets
- `model_pth/`: Pretrained model weights

## Citation

The paper is under review, : ) .

## License

This project is licensed under the MIT License.

---

For questions or issues, please open an [issue](https://github.com/aimanmutasem/MS-GBANet/issues).

