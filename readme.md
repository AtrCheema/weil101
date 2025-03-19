
Code for the paper [Machine learning analysis to interpret the effect of the photocatalytic reaction rate constant (k) of semiconductor-based photocatalysts on dye removal]( https://doi.org/10.1016/j.jhazmat.2023.132995) in Journal of Hazardous Materials.

# AI for Photocatalysis
In this study we performed data-driven modeling of photocatalysis process. The objective was to build a machine learning (ML) model to predict first order rate constant k using the experimental conditions (Time, solution pH, Light intensity, Light source distance, dye concentration loading), elemental composition of catalyst (C, Fe, Al, Ni, Mo, S, Bi, Ag, Pd, Pt) physio-chemical properties of the catalyst (Volume, surface area, pore size, pore volume) and parameters of pollutant (solubility, molecular weight, H-bond acceptor and donor counts). Total data consisted of 1527 samples and 32 features, which were collected by experimentation. This dataset was divided into 1068 (70%) training set and 459 (30%) test set. In the first notebook [Exploratory Data Analysis](https://weil101.readthedocs.io/en/latest/auto_examples/eda.htmly) we performed exploratory data analysis. After this we checked the performance of avaialble (over 30) machine learning algorithms on test set of our data in [Experiments](https://weil101.readthedocs.io/en/latest/auto_examples/experiments.html) after training them on training set. The purpose was to get an idea that which ML algorithm will be best for our problem. After that, we performed feature selection using various feature selection methods in [Feature Selection notebook](https://weil101.readthedocs.io/en/latest/auto_examples/brouta_feature_selection.html). The final features were selected using [Boruta-shap](https://doi.org/10.18637/jss.v036.i11) method. After selecting the algorithm and features, we performed hyperparameter optimization using k-fold cross validation in [hyperparameter optimization](https://weil101.readthedocs.io/en/latest/auto_examples/hpo.html). Then we built and trained our model on training set and checked its prediction performance on test set. Some plots depicting analysis of prediction performance and error anlaysis were also plotted here. After that we interpreted the machine learning model using various post-hoc interpretation methods. This includes [SHAP](https://arxiv.org/abs/1705.07874), [Partial Dependence Plots](https://hastie.su.domains/ElemStatLearn/) and [Accumulated Local Effects](https://doi.org/10.1111/rssb.12377). Finally we checked the robustness of our model by quantifying uncertainty in the prediction of machine learning model. We used [conformal analysis](https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf) for this purpose and analyzed the robustness of our model by employing various conformal anlaysis methods.

# Reproducibility
The results presented in these notebooks are completely (~100%) reproducible. All you need is to use same computational environment which was used to create these results. The names and versions of the python packages used in this project are given in requirements.txt file. Furthermore, the exact version of some of the python packages is also printed at the start of each notebook. The user will have to install these packages, preferably in a new conda environment. Then make sure that you have copied all the code in utils notebook in a utils.py file and saved in the same direcotry/folder where other python scripts are present. The data file is expected to be in the data folder. These steps can be summarized as below

    git clone https://github.com/AtrCheema/weil101.git

    cd weil101

    pip install -r docs/requirements.txt

    make html


Online reprducible examples running on readthedocs are at https://weil101.readthedocs.io/en/latest/ .

