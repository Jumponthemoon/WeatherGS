# 🌦 [ICRA 2025] WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project-Page-yellow)](https://jumponthemoon.github.io/weather-gs/)
[![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-blue)](https://arxiv.org/pdf/2412.18862)
![image](https://github.com/user-attachments/assets/8a23ba19-e259-4015-8cb6-d6ac8b970df2)

**WeatherGS** is a novel framework for 3D scene reconstruction under challenging weather conditions such as rain and snow. Built upon 3D Gaussian Splatting (3DGS), WeatherGS addresses the limitations of conventional 3DGS models when applied to scenes with dynamic, weather-induced noise.

## 🚀 Highlights

- 🌧️ **Weather-Resilient**: Effectively distinguishes and filters out transient artefacts to improve static scene reconstruction.
- 🧠 **Mask-Guided Optimization**: Leverages learned visibility and transient masks to suppress weather-induced distortions.
- 🎥 **Photo-Realistic Output**: Maintains high-fidelity rendering even under severe weather degradation.
- ⚡ **Fast Rendering**: Preserves the real-time rendering capability of the original 3D Gaussian Splatting framework.


## 📦 Installation

We recommend using Anaconda:

```bash
git clone https://github.com/Jumponthemoon/WeatherGS.git
cd WeatherGS
conda env create --file environment.yml
conda activate gaussian_splatting
```

Make sure your system supports PyTorch with GPU acceleration.

## 🏃‍♂️ Training


```bash
python train.py -s /path/to/scene --masks /path/to/scene/masks
```


## 📁 Datasets

Please download through this link:  
👉 **[Google Drive - WeatherGS Resource](https://drive.google.com/file/d/1S3fOnl-SEgiapFPm2s0VtUDeVYwdAnL_/view?usp=drive_link)**


## 📄 Citation

If you use this code or find this project helpful, please cite:

```bibtex
@misc{weathergs_qian,
      title={WeatherGS: 3D Scene Reconstruction in Adverse Weather Conditions via Gaussian Splatting}, 
      author={Chenghao Qian and Yuhu Guo and Wenjing Li and Gustav Markkula},
      year={2025},
      eprint={2412.18862},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.18862}, 
}
```

## 🤝 Acknowledgements

This work builds upon [3D Gaussian Splatting](https://repo-link) and benefits from open-source contributions in 3D vision, rendering, and weather simulation.
