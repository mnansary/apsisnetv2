conda install cudatoolkit=11.8.0 
conda install cudnn=8.9.2.26

pip install tensorflow-gpu==2.8.0
pip uninstall protobuf
pip install --no-binary protobuf protobuf==3.18.0
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.23.0
pip install matplotlib==3.9.2
pip install tqdm==4.66.5
pip install pandas==2.2.2
pip install opencv-python==4.10.0.84
pip install segmentation-models==1.0.1

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

