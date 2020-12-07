# UCCA 4 BPM

## Minimal Installation for Reproducing Our Results
install Miniconda: https://docs.conda.io/en/latest/miniconda.html

create new venv with `conda create --name ucca4bpm` and activate it `conda activate ucca4bpm`

run `conda config --add channels anaconda`

run `conda config --add channels conda-forge`

run `conda install python=3.6.11 gensim=3.8.3 matplotlib=3.3.3 scikit-learn=0.20.1`

run `pip install dgl==0.5.2 tensorflow==2.3.1`

## Reproducing Our Results

Download the prepared data and copy them into `ucca4bpm/data/transformed`, so that e.g. file `ours_qian_srl_google300.pickle` is in ucca4bpm/data/transformed. https://drive.google.com/drive/folders/1jumtVxkOAswTOktmTxQ1ode33kf3MuTV?usp=sharing

We have prepared several scripts for generating our runs and plots.

`ucca4bpm/experiments_ours.py` will run the analysis of hyper parameters and features.

`ucca4bpm/experiments_qian.py` will run our model on the data by Qian et al and on our dataset annotated by MGTC schema.

`ucca4bpm/experiments_quishpi.py` will run our model on the data by Qian et al and on our dataset annotated by ATDP schema.

`ucca4bpm/visualize_experiments.py` will run the visualization of our analysis runs.

`ucca4bpm/report.py` will print the reported runs to console.

You can run all of those by `python -m ucca4bpm.<your_script>` when you are in the top level directory.
