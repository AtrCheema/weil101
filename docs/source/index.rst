.. ai4photocatalysis documentation master file, created by
   sphinx-quickstart on Thu Mar 30 17:31:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AI for  Photocatalysis
=========================
In this study we performed data-driven modeling of photocatalysis process. The objective was
to build a machine learning (ML) model to predict first order rate constant ``k`` using the
experimental conditions (Time, solution pH, Light intensity, Light source distance, dye concentration
loading), elemental composition of catalyst (C, Fe, Al, Ni, Mo, S, Bi, Ag, Pd, Pt)
physio-chemical properties of the catalyst (Volume, surface area, pore size, pore volume)
and parameters of pollutant (solubility, molecular weight, H-bond acceptor and donor counts).
Total data consisted of 1527 samples
and 32 features, which were collected by experimentation. This dataset was divided into 1068 (70%) training
set and 459 (30%) test set. In the first notebook (:ref:`sphx_glr_auto_examples_eda.py`) we performed exploratory
data analysis. After this we checked the performance of avaialble (over 30) machine learning algorithms
on test set of our data in :ref:`sphx_glr_auto_examples_experiments.py` after training them on training set.
The purpose was to get an idea that which ML algorithm will be best for our problem.
After that, we performed feature selection using various feature selection methods
in :ref:`sphx_glr_auto_examples_brouta_feature_selection.py` notebook. The final features
were selected using Boruta-shap method. After selecting
the algorithm and features, we performed hyperparameter optimization using k-fold cross validation
in :ref:`sphx_glr_auto_examples_hpo.py`.
Then we built and trained our model on training set and checked its prediction performance on test set. Some plots
depicting analysis of prediction performance and error anlaysis were also plotted here. After
that we interpreted the machine learning model using various post-hoc interpretation methods. This
includes `SHAP <https://arxiv.org/abs/1705.07874>`_, `Partial Dependence Plots <https://hastie.su.domains/ElemStatLearn/>`_
and `Accumulated Local Effects <https://doi.org/10.1111/rssb.12377>`_. Finally we checked the robustness
of our model by quantifying uncertainty in the prediction of machine learning model. We used
`conformal analysis <https://jmlr.csail.mit.edu/papers/volume9/shafer08a/shafer08a.pdf>`_ for this purpose and
analyzed the robustness of our model by employing various conformal anlaysis methods.

Reproducibility
==================
The results presented in these notebooks are completely (~100%) reproducible. All you need is
to use same computational environment which was used to create these results. The names and
versions of the python packages used in this project are given in requirements.txt file.
Furthermore, the exact version of some of the python packages is also printed at the start of
each notebook. The user will have to install these packages, preferably in a new conda environment.
Then make sure that you have copied all the code in :ref:`sphx_glr_auto_examples_utils.py` notebook in a utils.py file and saved in the
same direcotry/folder where other python scripts are present. The data file is expected to be
in the ``data`` folder.


.. toctree::
   :maxdepth: 4

   auto_examples/index

