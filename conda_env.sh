conda create -n gaussian-avatars python=3.10 -y
conda activate gaussian-avatars
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8.0
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate
conda activate gaussian-avatars
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
pip install -r requirements.txt
