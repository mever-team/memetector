import argparse
import numpy as np
import os
from detectron2.config import get_cfg
from predictor import VisualizationDemo
import matplotlib.pyplot as plt
from PIL import Image
import time


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    base_dir = "./TextFuseNet"
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default=base_dir + "/configs/ocr/icdar2013_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default=base_dir + "/out_dir_r101/icdar2013_model/model_ic13_r101.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--input",
        default=base_dir + "/input_images/*.jpg",
        nargs="+",
        help="the folder of icdar2013 test images",
    )

    parser.add_argument(
        "--output",
        default=base_dir + "/test_icdar2013/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def keep_visual_part(bbs, h, w, preds):
    visual_part = False
    bbs = [
        [int(x) for x in bb.cpu().numpy()] for i, bb in enumerate(bbs) if preds[i] == 0
    ]
    rectangle = []
    for p in np.linspace(
        0.1, 0.9, 17
    ):  # percentage of the image area to keep for the center rectangle
        for ratio in np.linspace(np.sqrt(p), 1 / np.sqrt(p), 10):
            for wfrac in np.linspace(
                np.sqrt(p) / (2 * ratio), 1 - np.sqrt(p) / (2 * ratio), 10
            ):
                for hfrac in np.linspace(
                    ratio * np.sqrt(p) / 2, 1 - ratio * np.sqrt(p) / 2, 10
                ):
                    coordinates = [
                        w * wfrac - w * np.sqrt(p) / ratio / 2,
                        h * hfrac - h * np.sqrt(p) * ratio / 2,
                        w * wfrac + w * np.sqrt(p) / ratio / 2,
                        h * hfrac + h * np.sqrt(p) * ratio / 2,
                    ]
                    rectangle.append((coordinates, p, overlap(coordinates, bbs)))
    tmp = [x for x in rectangle if not x[2]]
    if tmp:
        maxp = max([x[1] for x in tmp])
        tmp = [x for x in tmp if x[1] == maxp]
        visual_part, area, contains_txt = tmp[np.random.choice(len(tmp))]
        visual_part = [int(x) for x in visual_part]
    return visual_part


def overlap(rec, bbs):
    for bb in bbs:
        if rec[0] < bb[2] and rec[2] > bb[0] and rec[1] < bb[3] and rec[3] > bb[1]:
            return True
    return False


args = get_parser().parse_args()
cfg = setup_cfg(args)
detection_demo = VisualizationDemo(cfg)
imgdir = "./data/meme"
cropdir = "./data/cropped/"
files = [
    os.path.join(imgdir, x) for x in os.listdir(imgdir) if (".jpg" in x or ".png" in x)
]
n = len(files)
start = time.time()
counter = 0
p_sum = 0
p_sum_2 = 0
for i, file in enumerate(files):
    filename = os.path.basename(file)
    savpath = f"{cropdir}cropped_{filename}"

    img = plt.imread(file)
    if img.shape[-1] == 3:
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Find text bounding boxes
        prediction, vis_output, polygons = detection_demo.run_on_image(img)
        boxes = prediction["instances"].pred_boxes
        height, width = prediction["instances"].image_size
        preds = prediction["instances"].pred_classes.cpu().numpy()

        # Visual part extraction
        visual_box = keep_visual_part(boxes, height, width, preds)

        # If text is detected in the image
        if visual_box:
            # Crop the visual part
            crop_img = img[visual_box[1] : visual_box[3], visual_box[0] : visual_box[2]]
            Image.fromarray(crop_img).save(savpath)

            # Calculate the mean and std fraction of the visual part area
            A_init = height * width
            A_R = (visual_box[2] - visual_box[0]) * (visual_box[3] - visual_box[1])
            percentage = A_R / A_init
            p_sum += percentage
            p_sum_2 += percentage**2
            counter += 1
            mean = p_sum / counter
            std = np.sqrt(p_sum_2 / counter - mean**2)

            elapsed = time.time() - start
            print(
                f"\r{i + 1}/{n}: {filename}, {counter}] p={percentage * 100:1.1f}%, p_avg={mean * 100:1.1f}%, p_std={std * 100:1.3f}%, {elapsed / 3600: 1.3f} h, ETA: {elapsed / (i + 1) * (n - i - 1) / 3600: 1.3f} h",
                end="",
            )

# Last print
# 10000/10000: 19845.png, 9984] p=35.1%, p_avg=64.3%, p_std=16.679%,  1.824 h, ETA:  0.000 h
