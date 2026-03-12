import os
from model.network_module import ParametersClassifier
from PIL import Image
from train_config import *
import time

sample_data = "../data/cropped_test/"

model = ParametersClassifier.load_from_checkpoint(
    # checkpoint_path=r"C:\Users\Alex\OneDrive - University of Hertfordshire\Year3\Final Year AI Project\Main\clean_src\sample_checkpoints\sample249.ckpt",
    # checkpoint_path=r"C:\Users\Alex\OneDrive - University of Hertfordshire\Year3\Final Year AI Project\Main\clean_src\checkpoints\04022026\1234\MHResAttNet-dataset_full-04022026-epoch=49-val_loss=0.02-val_acc=1.00.ckpt",
    checkpoint_path=r"C:\Users\Alex\OneDrive - University of Hertfordshire\Year3\Final Year AI Project\Main\clean_src\checkpoints\04022026\1234\MHResAttNet-dataset_full-04022026-epoch=15-val_loss=1.75-val_acc=0.76.ckpt",
    num_classes=3,
    gpus=0,
)
model.eval()

img_paths = [
    os.path.join(sample_data, img)
    for img in os.listdir(sample_data)
    if os.path.splitext(img)[1] in (".jpg", ".png")
]

print("********* CAXTON sample predictions *********")
print("Flow rate | Lateral speed | Z offset | Hotend")
print("*********************************************")

t1 = time.time()

for img_path in img_paths:
    pil_img = Image.open(img_path)
    x = preprocess(pil_img).unsqueeze(0)
    y_hats = model(x)
    # print(y_hats)
    y_hat0, y_hat1, y_hat2, y_hat3 = y_hats

    _, preds0 = torch.max(y_hat0, 1)
    _, preds1 = torch.max(y_hat1, 1)
    _, preds2 = torch.max(y_hat2, 1)
    _, preds3 = torch.max(y_hat3, 1)
    preds = torch.stack((preds0, preds1, preds2, preds3)).squeeze()

    preds_str = str(preds.numpy())
    img_basename = os.path.basename(img_path)
    print("Input:", img_basename, "->", "Prediction:", preds_str)

t2 = time.time()
print(f"Completed {len(img_paths)} predictions in {t2 - t1:.2f}s")