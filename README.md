# Drag3D: DragGAN meets GET3D

This project extends the idea of [DragGAN](https://github.com/XingangPan/DragGAN) into the [GET3D](https://github.com/nv-tlabs/GET3D) to enable interactive generation and drag editing of textured meshes.

We also implement a GUI to demonstrate real-time 3D point drag editing of 3D textured meshes:

**TODO: demo video**

## Install
```bash
# download
git clone https://github.com/ashawkey/Drag3D.git
cd Drag3D

# dependency
pip install -r requirements.txt

# (optional) get a better font to display
wget https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
```

Download pretrained GET3D checkpoints from [here](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW?usp=sharing) and put them under `./pretrained_model`.


### Tested Environment
The required GPU memory is about 4 GB.
* Ubuntu 20 + V100 + CUDA 11.6 + torch 1.12.0
* Windows 10 + 3070 + CUDA 12.1 + torch 2.1.0

## Usage
```bash
# run gui
python gui.py --outdir trial_car --resume_pretrain pretrained_model/shapenet_car.pt
```

You need to click `get` to generate a 3D model first.

Then, operate the GUI by:
* Left drag: rotate camera.
* Middle drag: pan camera.
* Scroll: scale camera.
* Right click: add / select source point.
* Right drag: drag target point.

After adding at least one point pair, click `train` to start optimization.

You can repeat these steps until get satisfying shapes.

Finally, click `save` to export textured mesh.



## Acknowledgement

* [DragGAN](https://github.com/XingangPan/DragGAN):
  ```latex
  @inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold}, 
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
  }
  ```

* [GET3D](https://github.com/nv-tlabs/GET3D):
  ```latex
  @inproceedings{gao2022get3d,
    title={GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images},
    author={Jun Gao and Tianchang Shen and Zian Wang and Wenzheng Chen and Kangxue Yin
    and Daiqing Li and Or Litany and Zan Gojcic and Sanja Fidler},
    booktitle={Advances In Neural Information Processing Systems},
    year={2022}
  }
  ```