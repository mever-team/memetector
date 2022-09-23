import numpy as np
from split import split
import tensorflow
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import AdamW
from vit import Patches, PatchEncoder, PositionalEmbedding
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

vgg = np.load("results/accuracy VGG16.npy")
rnet = np.load("results/accuracy ResNet50.npy")
eff = np.load("results/accuracy EfficientNetB5.npy")
vit = np.load("results/accuracy ViT.npy")
vita = np.load("results/accuracy ViTa.npy")

print("Table 2:")
print((100 * vita).round(1))

print("\nTable 3:")
print(
    f"VGG16 w/o VPU:\t\t\t",
    (np.max(vgg[9:, 9:], axis=0) * 100).round(2),
    (np.mean(vgg[9:, 9:]) * 100).round(2),
)
print(
    f"ResNet50 w/o VPU:\t\t",
    (np.max(rnet[9:, 9:], axis=0) * 100).round(2),
    (np.mean(rnet[9:, 9:]) * 100).round(2),
)
print(
    f"EfficientNetB5 w/o VPU:\t",
    (np.max(eff[9:, 9:], axis=0) * 100).round(2),
    (np.mean(eff[9:, 9:]) * 100).round(2),
)
print(
    f"ViT w/o VPU:\t\t\t",
    (np.max(vit[9:, 9:], axis=0) * 100).round(2),
    (np.mean(vit[9:, 9:]) * 100).round(2),
)
print(
    f"ViTa (ours) w/o VPU:\t",
    (np.max(vita[9:, 9:], axis=0) * 100).round(2),
    (np.mean(vita[9:, 9:]) * 100).round(2),
)
print(
    f"ViTa (ours) w/ VPU:\t\t",
    (vita[8, -4:] * 100).round(2),
    (np.mean(vita[8, -4:]) * 100).round(2),
)


print("\nTable 4:")
print("VGG16", (100 * vgg).mean().round(2))
print("ResNet", (100 * rnet).mean().round(2))
print("EfficientNetB5", (100 * eff).mean().round(2))
print("ViT", (100 * vit).mean().round(2))
print("MemeTector (ours)", (100 * vita).mean().round(2))

print(f"\nTable 5:")
n = vgg.shape[0] * vgg.shape[1]
print(
    f"VGG16: {np.sum(vita > vgg)}/{n}={np.sum(vita > vgg) / n * 100:1.2f}%. Average difference: {(np.mean(vita) - np.mean(vgg)) * 100: 1.2f}%"
)
print(
    f"ResNet50: {np.sum(vita > rnet)}/{n}={np.sum(vita > rnet) / n * 100:1.2f}%. Average difference: {(np.mean(vita) - np.mean(rnet)) * 100: 1.2f}%"
)
print(
    f"EfficientNetB5: {np.sum(vita > eff)}/{n}={np.sum(vita > eff) / n * 100:1.2f}%. Average difference: {(np.mean(vita) - np.mean(eff)) * 100: 1.2f}%"
)
print(
    f"ViT: {np.sum(vita > vit)}/{n}={np.sum(vita > vit) / n * 100:1.2f}%. Average difference: {(np.mean(vita) - np.mean(vit)) * 100: 1.2f}%"
)

# Figure 4: Attention plots
print("\nFigure 4...")
scenario = "ConcepCap 0.67 TXT 1.00"  # MemeTector: ConcepCap 0.67 TXT 1.00
category = {0: "regular", 1: "meme"}
transparency = 0.7  # 0.4
_, _, TEST = split(showtxt=False)

memes = [
    "./data/meme/96231.png",
    "./data/meme/95048.png",
    "./data/meme/98103.png",
    "./data/meme/98642.png",
    "./data/meme/94678.png",
    "./data/meme/90875.png",
    "./data/meme/94560.png",
    "./data/meme/93468.png",
    "./data/meme/90632.png",
    "./data/meme/93280.png",
]
regulars_wotxt = [
    "./data/web/text_absence/img_3092410.jpg",
    "./data/web/text_absence/img_3143070.jpg",
    "./data/web/text_absence/img_2629917.jpg",
    "./data/web/text_absence/img_3179606.jpg",
    "./data/web/text_absence/img_0906273.jpg",
]
regulars_wtxt = [
    "./data/web/text_presence/img_1380072.jpg",
    "./data/web/text_presence/img_1796174.jpg",
    "./data/web/text_presence/img_2225718.jpg",
    "./data/web/text_presence/img_1645037.jpg",
    "./data/web/text_presence/img_0559645.jpg",
]


ckpt = f"ckpt/{scenario} ViTa.h5"
model = tensorflow.keras.models.load_model(
    ckpt,
    custom_objects={
        "Patches": Patches,
        "PatchEncoder": PatchEncoder,
        "PositionalEmbedding": PositionalEmbedding,
        "adamw": AdamW,
    },
)
model = Model(
    inputs=model.input,
    outputs=[
        model.output,
        model.get_layer("tf.nn.softmax_136").output,
        model.get_layer("tf.nn.softmax_137").output,
        model.get_layer("tf.nn.softmax_138").output,
        model.get_layer("tf.nn.softmax_139").output,
    ],
)

plt.figure(figsize=(10, 7.9))

# Memes
for i, meme in enumerate(memes):
    imgshow = cv2.resize(
        np.array(Image.open(meme)), dsize=(250, 250), interpolation=cv2.INTER_CUBIC
    )
    imgpred = cv2.resize(
        (np.array(Image.open(meme)) / 255 - np.array([0.513, 0.485, 0.461]))
        / np.sqrt(np.array([0.107, 0.103, 0.108])),
        dsize=(250, 250),
        interpolation=cv2.INTER_CUBIC,
    )
    output = model.predict(np.expand_dims(imgpred, axis=0))
    attention = (output[1] + output[2] + output[3] + output[4]) / 4

    plt.subplot(4, 5, 1 + i)
    h = plt.imshow(imgshow)
    plt.imshow(
        np.resize(attention, (10, 10)),
        cmap="jet",
        alpha=transparency,
        extent=h.get_extent(),
        interpolation="bilinear",
    )
    plt.axis("off")

# Web - text absence
for i, regular in enumerate(regulars_wotxt):
    imgshow = cv2.resize(
        np.array(Image.open(regular)), dsize=(250, 250), interpolation=cv2.INTER_CUBIC
    )
    imgpred = cv2.resize(
        (np.array(Image.open(regular)) / 255 - np.array([0.513, 0.485, 0.461]))
        / np.sqrt(np.array([0.107, 0.103, 0.108])),
        dsize=(250, 250),
        interpolation=cv2.INTER_CUBIC,
    )
    output = model.predict(np.expand_dims(imgpred, axis=0))
    attention = (output[1] + output[2] + output[3] + output[4]) / 4

    plt.subplot(4, 5, 11 + i)
    h = plt.imshow(imgshow)
    plt.imshow(
        np.resize(attention, (10, 10)),
        cmap="jet",
        alpha=transparency,
        extent=h.get_extent(),
        interpolation="bilinear",
    )
    plt.axis("off")

# Web - text presence
for i, regular in enumerate(regulars_wtxt):
    imgshow = cv2.resize(
        np.array(Image.open(regular)), dsize=(250, 250), interpolation=cv2.INTER_CUBIC
    )
    imgpred = cv2.resize(
        (np.array(Image.open(regular)) / 255 - np.array([0.513, 0.485, 0.461]))
        / np.sqrt(np.array([0.107, 0.103, 0.108])),
        dsize=(250, 250),
        interpolation=cv2.INTER_CUBIC,
    )
    output = model.predict(np.expand_dims(imgpred, axis=0))
    attention = (output[1] + output[2] + output[3] + output[4]) / 4

    plt.subplot(4, 5, 16 + i)
    h = plt.imshow(imgshow)
    plt.imshow(
        np.resize(attention, (10, 10)),
        cmap="jet",
        alpha=transparency,
        extent=h.get_extent(),
        interpolation="bilinear",
    )
    plt.axis("off")

plt.tight_layout(pad=1.8, h_pad=0.1, w_pad=0.1)
plt.text(0.003, 0.82, "meme", rotation=90, fontsize=14, transform=plt.gcf().transFigure)
plt.text(0.003, 0.58, "meme", rotation=90, fontsize=14, transform=plt.gcf().transFigure)
plt.text(
    0.003,
    0.29,
    "regular w/o text",
    rotation=90,
    fontsize=14,
    transform=plt.gcf().transFigure,
)
plt.text(
    0.003,
    0.065,
    "regular w/ text",
    rotation=90,
    fontsize=14,
    transform=plt.gcf().transFigure,
)
plt.savefig("results/Fig4.png")
