# 🦕 Grounding DINO 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-mscoco)](https://paperswithcode.com/sota/zero-shot-object-detection-on-mscoco?p=grounding-dino-marrying-dino-with-grounded) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/zero-shot-object-detection-on-odinw)](https://paperswithcode.com/sota/zero-shot-object-detection-on-odinw?p=grounding-dino-marrying-dino-with-grounded) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=grounding-dino-marrying-dino-with-grounded) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/grounding-dino-marrying-dino-with-grounded/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=grounding-dino-marrying-dino-with-grounded)


**[IDEA-CVR, IDEA-Research](https://github.com/IDEA-Research)** 

[Shilong Liu](http://www.lsl.zone/), [Zhaoyang Zeng](https://scholar.google.com/citations?user=U_cvvUwAAAAJ&hl=zh-CN&oi=ao), [Tianhe Ren](https://rentainhe.github.io/), [Feng Li](https://scholar.google.com/citations?user=ybRe9GcAAAAJ&hl=zh-CN), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=zh-CN), [Jie Yang](https://github.com/yangjie-cv), [Chunyuan Li](https://scholar.google.com/citations?user=Zd7WmXUAAAAJ&hl=zh-CN&oi=ao), [Jianwei Yang](https://jwyang.github.io/), [Hang Su](https://scholar.google.com/citations?hl=en&user=dxN1_X0AAAAJ&view_op=list_works&sortby=pubdate), [Jun Zhu](https://scholar.google.com/citations?hl=en&user=axsP38wAAAAJ), [Lei Zhang](https://www.leizhang.org/).


[[`Paper`](https://arxiv.org/abs/2303.05499)] [[`Demo`](https://huggingface.co/spaces/ShilongLiu/Grounding_DINO_demo)] [[`BibTex`](#black_nib-citation)]


PyTorch implementation and pretrained models for Grounding DINO. For details, see the paper **[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)**.

- 🔥 **[Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2)** is released now, which combines Grounding DINO with [SAM 2](https://github.com/facebookresearch/segment-anything-2) for any object tracking in open-world scenarios.
- 🔥 **[Grounding DINO 1.5](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)** is released now, which is IDEA Research's **Most Capable** Open-World Object Detection Model!
- 🔥 **[Grounding DINO](https://arxiv.org/abs/2303.05499)** and **[Grounded SAM](https://arxiv.org/abs/2401.14159)** are now supported in Huggingface. For more convenient use, you can refer to [this documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino)

## 💡 Highlight

- **Open-Set Detection.** Detect **everything** with language!
- **High Performance.** COCO zero-shot **52.5 AP** (training without COCO data!). COCO fine-tune **63.0 AP**.
- **Flexible.** Collaboration with Stable Diffusion for Image Editting.

<details open>
<summary><font size="4">
Description
</font></summary>
 <a href="https://arxiv.org/abs/2303.05499">Paper</a> introduction.
<img src=".asset/hero_figure.png" alt="ODinW" width="100%">
Marrying <a href="https://github.com/IDEA-Research/GroundingDINO">Grounding DINO</a> and <a href="https://github.com/gligen/GLIGEN">GLIGEN</a>
<img src="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GD_GLIGEN.png" alt="gd_gligen" width="100%">
</details>

## ⭐ Explanations/Tips for Grounding DINO Inputs and Outputs
- Grounding DINO accepts an `(image, text)` pair as inputs.
- It outputs `900` (by default) object boxes. Each box has similarity scores across all input words. (as shown in Figures below.)
- We defaultly choose the boxes whose highest similarities are higher than a `box_threshold`.
- We extract the words whose similarities are higher than the `text_threshold` as predicted labels.
- If you want to obtain objects of specific phrases, like the `dogs` in the sentence `two dogs with a stick.`, you can select the boxes with highest text similarities with `dogs` as final outputs. 
- Note that each word can be split to **more than one** tokens with different tokenlizers. The number of words in a sentence may not equal to the number of text tokens.
- We suggest separating different category names with `.` for Grounding DINO.
![model_explain1](.asset/model_explan1.PNG)
![model_explain2](.asset/model_explan2.PNG)

## 🛠️ Install 

**Note:**

0. If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

Please make sure following the installation steps strictly, otherwise the program may produce: 
```bash
NameError: name '_C' is not defined
```

If this happened, please reinstalled the groundingDINO by reclone the git and do all the installation steps again.
 
#### how to check cuda:
```bash
echo $CUDA_HOME
```
If it print nothing, then it means you haven't set up the path/

Run this so the environment variable will be set under current shell. 
```bash
export CUDA_HOME=/path/to/cuda-11.3
```

Notice the version of cuda should be aligned with your CUDA runtime, for there might exists multiple cuda at the same time. 

If you want to set the CUDA_HOME permanently, store it using:

```bash
echo 'export CUDA_HOME=/path/to/cuda' >> ~/.bashrc
```
after that, source the bashrc file and check CUDA_HOME:
```bash
source ~/.bashrc
echo $CUDA_HOME
```

In this example, /path/to/cuda-11.3 should be replaced with the path where your CUDA toolkit is installed. You can find this by typing **which nvcc** in your terminal:

For instance, 
if the output is /usr/local/cuda/bin/nvcc, then:
```bash
export CUDA_HOME=/usr/local/cuda
```
**Installation and Setup:**

Note that experiments were run using Python 3.9.

1. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
```

2. Install the required dependencies in the current directory.

```bash
pip install -e . --no-build-isolation
pip install -r requirements.txt
```

3. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

4. Download the dataset.

```bash
cd experiments
python3 download_dataset.py
```

5. Run experiments.

```bash
python3 run_experiments.py
```

5. Run notebook `experiments/expierments.ipynb`.

## ❤️ Acknowledgement

Our model is related to [DINO](https://github.com/IDEA-Research/DINO) and [GLIP](https://github.com/microsoft/GLIP). Thanks for their great work!

We also thank great previous work including DETR, Deformable DETR, SMCA, Conditional DETR, Anchor DETR, Dynamic DETR, DAB-DETR, DN-DETR, etc. More related work are available at [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer). A new toolbox [detrex](https://github.com/IDEA-Research/detrex) is available as well.

Thanks [Stable Diffusion](https://github.com/Stability-AI/StableDiffusion) and [GLIGEN](https://github.com/gligen/GLIGEN) for their awesome models.


## ✒️ Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```




