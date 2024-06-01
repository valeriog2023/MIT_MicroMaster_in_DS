
# README.md

### These are notes from the course **Machine Learning with Python-From Linear Models to Deep Learning** from MIT available on EDX:

https://learning.edx.org/course/course-v1:MITx+6.86x+2T2024/home

## Required python packages

Throughout this course, we will be using Python 3.8 along with the following packages. Code written in new versions of python will be accepted, as long as functions/features that are available only in Python 3.9 or beyond are not used.
- NumPy
- matplotlib
- scikit-learn
- SciPy
- tqdm
- PyTorch

For PyTorch, follow the instructions on https://pytorch.org/to install from pip repository corresponding to your system. You will not need CUDA for this course.

```
python3 -m venv ML
source ML/bin/activate
pip3 install --upgrade pip
pip3 install numpy matplotlib scipy tqdm scikit-learn
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

deactivate
```

### Anaconda
Anaconda is combination of :
- tool to manage virtual environment
- collection of packages used for DS and ML

```
! just for me.. to update base anaconda setup
conda update -n base -c defaults conda
!
conda create --name ML python=3.8
conda env list
conda activate ML
!
conda install  numpy matplotlib scipy tqdm scikit-learn
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install typing-extensions

conda deactivate
!
! if you watnt to remove the venv
conda env remove --name ML
```