# read .env file
set -o allexport
[ -f .env ] && . .env
sudo apt update
sudo apt install gh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -u
conda env create -f environment.yml
conda activate twotower
git config --global user.email $GITHUB_EMAIL
git config --global user.name $GITHUB_USER