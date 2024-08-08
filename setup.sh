# read .env file
set -o allexport
[ -f .env ] && . .env
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -u
source ~/.bashrc
conda env create -f environment.yml
conda activate twotower