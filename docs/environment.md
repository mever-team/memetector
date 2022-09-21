# memetector environment
In order to run the files that use text detection (vpu.py, text.py),
you need to build the environment of TextFuseNet through the guidelines
in https://github.com/ying09/TextFuseNet. For the rest (model training & results)
you can use the following environment:
```
conda create -n memetector python=3.8
conda activate memetector
conda install cudatoolkit=11.0
conda install -c conda-forge cudnn=8.0
pip install tensorflow==2.7.0
pip install -U tensorflow-addons==0.17.1
pip install pillow==9.2.0
pip install pandas==1.5.0
pip install matplotlib==3.6.0
pip install opencv-python==4.6.0.66
```