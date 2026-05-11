<h2 align="center"> <a href="https://rover-xingyu.github.io/TTT3R">TTT3R: 3D Reconstruction as Test-Time Training</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2509.26645-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.26645) 
[![Home Page](https://img.shields.io/badge/Project-Website-33728E.svg)](https://rover-xingyu.github.io/TTT3R) 
![ICLR 2026](https://img.shields.io/badge/ICLR-2026-e28353.svg)
![China3DV 2026 Top 5 Paper](https://img.shields.io/badge/China3DV%202026-Top%205%20Paper-b8736c.svg)
[![X](https://img.shields.io/badge/@Xingyu%20Chen-black?logo=X)](https://x.com/RoverXingyu)  [![Bluesky](https://img.shields.io/badge/@Xingyu%20Chen-white?logo=Bluesky)](https://bsky.app/profile/xingyu-chen.bsky.social)


[Xingyu Chen](https://rover-xingyu.github.io/),
[Yue Chen](https://fanegg.github.io/),
[Yuliang Xiu](https://xiuyuliang.cn/),
[Andreas Geiger](https://www.cvlibs.net/),
[Anpei Chen](https://apchenstu.github.io/)
</h5>

<div align="center">
TL;DR: A simple state update rule to enhance length generalization for CUT3R.
</div>
<br>

https://github.com/user-attachments/assets/b7583837-1f1e-43a4-b281-09f340ee5918

## Getting Started

### Installation

1. Clone TTT3R.
```bash
git clone https://github.com/Inception3D/TTT3R.git
cd TTT3R
```

2. Create the environment.
```bash
conda create -n ttt3r python=3.11 cmake=3.14.0
conda activate ttt3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
# for evaluation
pip install evo
pip install open3d
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoints

CUT3R provide checkpoints trained on 4-64 views: [`cut3r_512_dpt_4_64.pth`](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link).

To download the weights, run the following commands:
```bash
cd src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```

### Inference Demo

To run the inference demo, you can use the following command:
```bash
# input can be a folder or a video
# the following script will run inference with TTT3R and visualize the output with viser on port 8080
CUDA_VISIBLE_DEVICES=6 python demo.py --model_path MODEL_PATH --size 512 \
    --seq_path SEQ_PATH --output_dir OUT_DIR --port 8080 \
    --model_update_type ttt3r --frame_interval 1 --reset_interval 100 \
    --downsample_factor 1000 --vis_threshold 5.0

# Example:
CUDA_VISIBLE_DEVICES=6 python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 \
    --seq_path examples/westlake.mp4 --output_dir tmp/taylor --port 8080 \
    --model_update_type ttt3r --frame_interval 1 --reset_interval 100 \
    --downsample_factor 100 --vis_threshold 6.0

CUDA_VISIBLE_DEVICES=6 python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 \
    --seq_path examples/taylor.mp4 --output_dir tmp/taylor --port 8080 \
    --model_update_type ttt3r --frame_interval 1 --reset_interval 50 \
    --downsample_factor 100 --vis_threshold 10.0 
```
Output results will be saved to `output_dir`.


### Evaluation
Please refer to the [eval.md](eval/eval.md) for more details.

## Acknowledgements
Our code is based on the following awesome repositories:

- [CUT3R](https://github.com/CUT3R/CUT3R)
- [Easi3R](https://github.com/Inception3D/Easi3R)
- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r.git)
- [Spann3R](https://github.com/HengyiWang/spann3r.git)
- [Viser](https://github.com/nerfstudio-project/viser)

We thank the authors for releasing their code!

## Citation

If you find our work useful, please cite:

```bibtex
@article{chen2025ttt3r,
    title={TTT3R: 3D Reconstruction as Test-Time Training},
    author={Chen, Xingyu and Chen, Yue and Xiu, Yuliang and Geiger, Andreas and Chen, Anpei},
    journal={arXiv preprint arXiv:2509.26645},
    year={2025}
    }
```
