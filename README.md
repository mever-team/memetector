# memetector
Implementation of meme detection method described in paper "MemeTector: Enforcing deep focus for meme detection".
The preprint can be found here: https://arxiv.org/abs/2205.13268

## Abstract
> Image memes and specifically their widely-known variation image macros, is a special new media type that combines text with images and is used in social media to playfully or subtly express humour, irony, sarcasm and even hate. It is important to accurately retrieve image memes from social media to better capture the cultural and social aspects of online phenomena and detect potential issues (hate-speech, disinformation). Essentially, the background image of an image macro is a regular image easily recognized as such by humans but cumbersome for the machine to do so due to feature map similarity with the complete image macro. Hence, accumulating suitable feature maps in such cases can lead to deep understanding of the notion of image memes. To this end, we propose a methodology that utilizes the visual part of image memes as instances of the regular image class and the initial image memes as instances of the image meme class to force the model to concentrate on the critical parts that characterize an image meme. Additionally, we employ a trainable attention mechanism on top of a standard ViT architecture to enhance the model's ability to focus on these critical parts and make the predictions interpretable. Several training and test scenarios involving web-scraped regular images of controlled text presence are considered in terms of model robustness and accuracy. The findings indicate that light visual part utilization combined with sufficient text presence during training provides the best and most robust model, surpassing state of the art.

![](https://github.com/mever-team/memetector/blob/main/docs/Figure%202.png)


## Guidelines
1. Download the memetector repository:
```
git clone https://github.com/mever-team/memetector.git
cd memetector
```
2. Download the [Facebook Hateful Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/) dataset 
from [here](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset)
and extract all the 10K images in data/meme folder
3. Setup two conda environments following [environment.md](docs/environment.md)
4. Run the [vpu.py](vpu.py) script in order to extract the visual parts of the memes of 
data/meme folder and store them in the data/cropped folder (using the text detection environment)
5. Download the randomly selected web-scraped images 
(from [Google's Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) dataset)
using their URLs in the files [urls_text_absence.csv](data/urls_text_absence.csv) and
[urls_text_presence.csv](data/urls_text_presence.csv) (note that keeping the same filenames is crucial
for the creation of identical spilts)
and place them in the corresponding subfolders of data/web folder
(text_absence and text_presence). - *These files have been selected from the whole dataset
by the use of text.py*
6. For reproducing the experiments run the train.py file
7. For reproducing the results presented in the paper run the results.py file

## Citation
If you use this code for your research, please cite our paper.
```
@article{koutlis2022memetector,
  title     = {MemeTector: Enforcing deep focus for meme detection},
  author    = {Koutlis, Christos and Schinas, Manos and Papadopoulos, Symeon},
  journal   = {arXiv preprint arXiv:2205.13268},
  year      = {2022},
}
```

## Acknowledgements
This work is partially funded by the
Horizon 2020 European project "MediaVerse: A universe of media assets and co-creation opportunities at your fingertips"
under grant agreement no. 957252.

## Licence
This project is licensed under the Apache License 2.0 - see the [LISENCE](LICENSE) file for details.

## Contact
Christos Koutlis ([ckoutlis@iti.gr](ckoutlis@iti.gr))
