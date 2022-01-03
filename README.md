# VRDL_LW4

```sh
python3 tools/gdget.py 1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb -O dataset.zip
unzip -qq dataset.zip -d ./
rm dataset.zip

pip install -r requirements.txt
```


```py
from torchsummary import summary
import torch
torch.cuda.is_available()
for GPU_ID in range(torch.cuda.device_count()):
	print(torch.cuda.get_device_name(GPU_ID))
```