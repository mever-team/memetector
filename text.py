import argparse
import numpy as np
import os
from detectron2.config import get_cfg
from predictor import VisualizationDemo
import matplotlib.pyplot as plt
import time
import pickle


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
    base_dir = "/home/ckoutlis/ModelStorage/TextFuseNet"
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


args = get_parser().parse_args()
cfg = setup_cfg(args)
detection_demo = VisualizationDemo(cfg)

maximum = len(os.listdir("./data/cropped"))
imgdir = "/path/to/google/conceptual/captions/dataset"
savwtxtpath = "./data/web/wtxt.pickle"
savwotxtpath = "./data/web/wotxt.pickle"
files_wtxt = []
files_wotxt = []

files = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
n = len(files)
start = time.time()
for i, file in enumerate(files):
    filename = os.path.basename(file)
    img = plt.imread(file)
    if img.shape[-1] == 3:
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        prediction, _, _ = detection_demo.run_on_image(img)

        if len(prediction["instances"]) > 0 and len(files_wtxt) < maximum:
            files_wtxt.append(file)
            with open(savwtxtpath, "wb") as handle:
                pickle.dump(files_wtxt, handle)
        elif len(prediction["instances"]) == 0 and len(files_wotxt) < maximum:
            files_wotxt.append(file)
            with open(savwotxtpath, "wb") as handle:
                pickle.dump(files_wotxt, handle)

    elapsed = time.time() - start
    print(
        f"{i + 1}/{n}: {filename}, w-txt:{len(files_wtxt)}/{maximum} - wo-txt: {len(files_wotxt)}/{maximum}, Elapsed: {elapsed / 3600: 1.3f} h"
    )

    if len(files_wtxt) == maximum and len(files_wotxt) == maximum:
        break
