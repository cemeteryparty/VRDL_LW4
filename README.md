# VRDL_LW4

## Reproduce Submission

Download model weight from [GDrive LINK](https://drive.google.com/drive/folders/1r8TQt90db4862RnQ3BzTzGFonnl8c0iV?usp=sharing)

Make sure to run below script after finishing [environment setting](#Installation).

```sh
git clone https://github.com/cemeteryparty/VRDL_LW4.git
cd VRDL_LW4/
python3 tools/gdget.py 1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O dataset.zip
unzip -qq dataset.zip -d ./
rm dataset.zip

mv training_hr_images/training_hr_images/* training_hr_images/
rm -rf training_hr_images/training_hr_images/
mv testing_lr_images/testing_lr_images/* testing_lr_images/
rm -rf testing_lr_images/testing_lr_images/

python3 inference.py --upscale_factor 4 --model-path netG_SRx4.pth
```

The 14 pred images will be generated under `testing_hr_images` directory.

## Installation

```sh
git clone https://github.com/cemeteryparty/VRDL_LW4.git
cd VRDL_LW4/
```

### Activate environment ###

```sh
conda create --name ENV_NAME python=3.7
conda activate ENV_NAME
```

### Install Library ###

```sh
pip install -r requirements.txt
```

#### Check GPU Support

```py
import torch
torch.cuda.is_available()
for GPU_ID in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(GPU_ID))
```

## Dataset Preparation

### Download dataset provided in GDrive

```sh
python3 tools/gdget.py 1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O dataset.zip
unzip -qq dataset.zip -d ./
rm dataset.zip

mv training_hr_images/training_hr_images/* training_hr_images/
rm -rf training_hr_images/training_hr_images/
mv testing_lr_images/testing_lr_images/* testing_lr_images/
rm -rf testing_lr_images/testing_lr_images/
```

## Train Process
```sh
python3 train.py --dataset-path training_hr_images --crop_size 320 --upscale_factor 4 \
    --epochs 100 --batch-size 8 --save-path models
```

## Inference

```sh
python3 inference.py --upscale_factor 4 --model-path models/netG_SRx4.pth
```

## Prediction Result

<p style="margin: auto">
    <img src="images/testing_lr_images/00.png" style="width: 40%">
    <img src="images/testing_hr_images/00_pred.png" style="width: 40%">
</p>

## Reference

[leftthomas/SRGAN](https://github.com/leftthomas/SRGAN)

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
