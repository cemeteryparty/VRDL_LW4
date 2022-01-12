from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from SRGAN.model import Generator
import numpy as np
from PIL import Image
import torch
import time
import cv2
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform LR to SR")
    parser.add_argument(
        "--upscale_factor", required=True, default=4, choices=[2, 4, 8],
        type=int, help="super resolution upscale factor"
    )
    parser.add_argument(
        "--model-path", required=True, default="models", type=str, help="model path"
    )
    args = parser.parse_args()
    # print(args)
    # exit(0)

    UPSCALE_FACTOR = args.upscale_factor
    TEST_MODE = "GPU"
    TEST_MODE = True if TEST_MODE == "GPU" else False
    MODEL_PATH = args.model_path

    model = Generator(UPSCALE_FACTOR).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
        )

    DATASET_PATH = "testing_lr_images"
    OUTPUT_PATH = "testing_hr_images"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    img_names = os.listdir(DATASET_PATH)
    for i in range(14):
        TimeStpBegin = time.time()
        lr_img = Image.open(os.path.join(DATASET_PATH, "{:02d}.png".format(i)))
        print("Inference image {:02d}.png with size={}.".format(i, lr_img.size))
        width, height = lr_img.size
        with torch.no_grad():
            image = Variable(ToTensor()(lr_img)).unsqueeze(0)
            if TEST_MODE:
                image = image.cuda()
            img_pred = model(image)
        img_pred = ToPILImage()(img_pred[0].data.cpu())
        # img_pred.save(os.path.join(OUTPUT_PATH, "{:02d}_pred_4x.png".format(i)))
        print("\tInference image to size={} for {:.2f} sec.".format(img_pred.size, time.time() - TimeStpBegin))

        # hr_img = img_pred.resize((width * 3, height * 3), resample=Image.BICUBIC)
        # print("\tResize to size={}.".format(hr_img.size))
        # hr_img.save(os.path.join(OUTPUT_PATH, "{:02d}_pred.png".format(i)))
        hr_img = cv2.resize(
            np.array(img_pred)[:, :, ::-1], (width * 3, height * 3),
            interpolation=cv2.INTER_AREA
        )  # cv2.INTER_CUBIC
        print("\tResize to size={}.".format(hr_img.shape))
        cv2.imwrite(os.path.join(OUTPUT_PATH, "{:02d}_pred.png".format(i)), hr_img)
