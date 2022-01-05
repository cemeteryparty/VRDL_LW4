from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from SRGAN.model import Generator
from PIL import Image
import torch
import time
import os


UPSCALE_FACTOR = 4
TEST_MODE = "GPU"
TEST_MODE = True if TEST_MODE == "GPU" else False
MODEL_PATH = "models/netG_SRx4.pth"

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    )

DATASET_PATH = "testing_lr_images"
OUTPUT_PATH =  "testing_hr_images"
os.makedirs(OUTPUT_PATH, exist_ok=True)
img_names = os.listdir(DATASET_PATH)
for i in range(14):
    TimeStpBegin = time.time()
    lr_img = Image.open(os.path.join(DATASET_PATH, "{:02d}.png".format(i)))
    width, height = lr_img.size
    image = Variable(ToTensor()(lr_img), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()
    img_pred = model(image)
    x4_img = ToPILImage()(img_pred[0].data.cpu())
    x4_img.save(os.path.join(OUTPUT_PATH, "{:02d}_pred_4x.png".format(i)))
    print(
        "Inference image {:02d}.png for {:.2f} sec.".format(i, time.time() - TimeStpBegin)
    )

    # hr_img = x4_img.resize((width * 3, height * 3), resample=Image.BICUBIC)
    # hr_img.save(os.path.join(OUTPUT_PATH, "{:02d}_pred.png".format(i)))
