# Not All Features Are Equal: Discovering Essential Features for Preserving Prediction Privacy
Code  for the WWW21 paper, [Cloak]{https://arxiv.org/abs/2003.12154}.


The code and model checkpoints used to produce the results are anonymously provided at \url{https://zenodo.org/record/4101695} and \url{https://zenodo.org/record/3887495}, respectively. The code is named \texttt{code.zip}  and the models and numpy files are named \texttt{saved\_nps.zip} and they both have the same directory structure. They each contain 5 Folders named \textit{exp1-trade-off, exp2-adversary, exp3-black-box, exp4-fairness} and \textit{exp5-shredder} which are related to the results in the experiments sectoin in the same order. The pre-trained parameters needed are provided in the  \texttt{saved\_nps.zip}, in the corresponding directory. So, all that is needed to be done is to copy all files from the \texttt{saved\_nps.zip
} directory to their corresponding positions in the code folders, and then run the provided Jupyter notebooks. 

For acquiring the datasets, you can have a look at the \texttt{acquire\_datasets.ipynb} notebook, included in the \texttt{code.zip}.


In short, each notebook  has \sieve in its name will start by loading the required datasets and then creating a model. Then, the model is trained based on the experiments and using the hyperparameters provided in section~\ref{sec:hyper}.
%
Finally, you can run a test function that is provided to evaluate the model. 
%For Experiment2, at the end of the training notebooks there is also an script that generates the original and noisy representations for mutual information estimation.
For seeing how the mutual information is estimated, you can run the notebooks that have \texttt{mutual\_info} in their names.
%
You need not have run the training before hand, if you place the provided \texttt{.npy} files in the correct directories. For the mutual information estimation you will need to download the ITE toolbox~\cite{itetoolbox}. The link is provided in the code. 
%%%%%%%%%%%%%%


