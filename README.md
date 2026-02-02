# 🌦 [ICRA 2025] WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project-Page-yellow)](https://jumponthemoon.github.io/weather-gs/)
[![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-green)](https://arxiv.org/pdf/2412.18862)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Jumponthemoon/WeatherGS)

![image](https://github.com/user-attachments/assets/8a23ba19-e259-4015-8cb6-d6ac8b970df2)

**WeatherGS** is a novel framework for 3D scene reconstruction under challenging weather conditions such as rain and snow. Built upon 3D Gaussian Splatting (3DGS), WeatherGS addresses the limitations of conventional 3DGS models when applied to scenes with dynamic, weather-induced noise.

## 🚀 Highlights
> ⭐ **If you like our work or find it useful, please give us a star or cite below. Thanks!**

- 🌧️ **Weather-Resilient**: Effectively distinguishes and filters out transient artefacts to improve static scene reconstruction.
- 🧠 **Mask-Guided Optimization**: Leverages learned visibility and transient masks to suppress weather-induced distortions.
- 🎥 **Photo-Realistic Output**: Maintains high-fidelity rendering even under severe weather degradation.
- ⚡ **Fast Rendering**: Preserves the real-time rendering capability of the original 3D Gaussian Splatting framework.


Make sure your system supports PyTorch with GPU acceleration.

## 🧰 Clone Code
```bash
git clone https://github.com/Jumponthemoon/WeatherGS.git
cd WeatherGS
```
## 📦 Installation

We recommend using Anaconda:

```bash
conda env create --file environment.yml
conda activate weathergs
```
## 📁 Datasets

You can download preprocessed dataset through this link:  
👉 **[Google Drive - WeatherGS Resource](https://drive.google.com/file/d/1S3fOnl-SEgiapFPm2s0VtUDeVYwdAnL_/view?usp=drive_link)**

## 🏃‍♂️ Training

```bash
cd 3DGS
python train.py -s /path/to/scene --masks /path/to/scene/masks
```

### 🔧 (Optional) Preprocessing Your Own Scenes

Due incompatibility, you should build **another two seperated environments** to run the **AEF** and **LED** modules:

#### AEF (Atmospheric Effect Filter)

```bash
cd AEF
conda env create --file environment.yml
conda activate aef
python infer.py --image_path /path/to/images
```

If you know the type of the scene, you can add --task for better performance:

```bash
python infer.py --image_path /path/to/images --task desnow  # or derain
```

#### LED (Len Effect Detector)

> ⚠ Requires TensorFlow 1.15. We recommend using a separate environment:

```bash
cd LED
conda deactivate
conda create -n led python=3.6
conda activate led
pip install -r requirements.txt
```

First generate coarse masks:
```bash
python detect_occlusion.py --image_path /path/to/processed_images/
```
Then refine the mask by changing threshold and dilation parameters:
```bash
python refine_mask.py --scene_path /path/to/scene/
```
The final masks will be saved in the `masks/` directory.


## 📄 Citation

If you use this code or find this project helpful, please cite:

```bibtex
@INPROCEEDINGS{11128699,
  author={Qian, Chenghao and Guo, Yuhu and Li, Wenjing and Markkula, Gustav},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Weathergs: 3D Scene Reconstruction in Adverse Weather Conditions Via Gaussian Splatting}, 
  year={2025},
  pages={185-191},
  keywords={Three-dimensional displays;Precipitation;Detectors;Benchmark testing;Light emitting diodes;Cameras;Robotics and automation;Image reconstruction;Meteorology;Lenses},
  doi={10.1109/ICRA55743.2025.11128699}}

```

## 🤝 Acknowledgements

This work builds upon [3D Gaussian Splatting](https://repo-link), [DiffPlugin](https://github.com/yuhaoliu7456/Diff-Plugin) and [AttGAN](https://github.com/MaybeShewill-CV/attentive-gan-derainnet). We thank for their amazing works!
