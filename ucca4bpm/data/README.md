# UCCA 4 BPM

## Minimal Installation for Reproducing Our Results
install conda
create new venv
run `conda install python=3.6.11`
run `conda config --add channels anaconda`
run `conda config --add channels conda-forge`
run `conda install dgl=0.5.2 gensim=3.8.3 tensorflow=2.3.1 matplotlib=3.3.3`

## Reproducing Our Results

We have prepared several scripts for generating our runs and plots.

`ucca4bpm/experiments_ours.py` will run the analysis of hyper parameters and features.
`ucca4bpm/experiments_qian.py` will run our model on the data by Qian et al and on our dataset annotated by MGTC schema.
`ucca4bpm/experiments_quishpi.py` will run our model on the data by Qian et al and on our dataset annotated by ATDP schema.

`ucca4bpm/visualize_experiments.py` will run the visualization of our analysis runs.

`ucca4bpm/report.py` will print the reported runs to console.