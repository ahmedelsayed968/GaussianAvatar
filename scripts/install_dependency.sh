conda create -n gs-avatar python=3.9 && conda activate gs-avatar

# install pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c iopath iopath
conda install -c bottler nvidiacub

# before installing pytorch3d
pip install --upgrade pip
pip install fvcore

# now install pytorch3d safely
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html

pip install torchaudio
pip install tqdm tensorboard scipy lpips open3d trimesh opencv-python
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
pip install PyOpenGL PyOpenGL_accelerate glfw