# Not All Features Are Equal: Discovering Essential Features for Preserving Prediction Privacy
Code  for the WWW21 paper, [Cloak](https://arxiv.org/abs/2003.12154).


The model checkpoints used to produce the results are provided at \url{https://zenodo.org/record/3887495}. 

All the code is available in the form of Jupyter notebooks, in the *code* directory in this repo. You need to first download all the checkpoints and numpy folders from the link above, and then extract them and place them in their corrosponding directories in the code folder.The code and numpy files have the same directory structure. They each contain 5 Folders named *exp1-trade-off, exp2-adversary, exp3-black-box, exp4-fairness and exp5-shredder* which are related to the results in the experiments sectoin in the same order. 

For acquiring the datasets, you can have a look at the \texttt{acquire\_datasets.ipynb} notebook, available in the code directory.


In short, each notebook  that has cloak in its name will start by loading the required datasets and then creating a model. Then, the model is trained based on the experiments and using the hyperparameters provided in the paper's appendix.

Finally, you can run a test function that is provided to evaluate the model. 

For Experiment2, at the end of the training notebooks there is also an script that generates the original and noisy representations for mutual information estimation.

For seeing how the mutual information is estimated, you can run the notebooks that have *mutual_info* in their names.

You need not have run the training before hand, if you place the provided *.npy* files in the correct directories. For the mutual information estimation you will need to download the ITE toolbox. The link is provided in the code. 

#### If you have any questions, please email Fatemeh Mireshghallah at fatemeh@ucsd.edu.

# Citation

If you use our code or find the material in the paper helpful, please cite this work using the following bib entry:

@inproceedings{www2021-cloak-mireshghallah, 
title={Not All Features Are Equal: Discovering Essential Features for Preserving Prediction Privacy}, 
author={Fatemehsadat Mireshghallah and Mohammadkazem Taram and Ali Jalali and Ahmed Taha Elthakeb and Dean Tullsen and Hadi Esmaeilzadeh}, 
booktitle={Proceedings of The Web Conference 2021},
Month = {April},
year={2021}  , 
}
