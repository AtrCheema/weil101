"""
================================
5. hyperparameter optimization
================================
Now that we have selected the model and input features. Now we will
try to improve the prediction performance of our model using hyperparameter
optimization.
"""

import os
from typing import Union

import numpy as np

import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.utils.utils import jsonize
from ai4water.utils.utils import TrainTestSplit, dateandtime_now
from ai4water.hyperopt import Categorical, Real, Integer, HyperOpt

from utils import prepare_data, set_rcParams, SAVE, version_info

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%
inputs = ['Solution pH', 'Time (m)', 'Anions', 'Ni (At%)', 'HA (mg/L)',
          'loading (g)', 'Pore size (nm)', 'O (At%)',
          'Light intensity (watt)', 'Mo (At%)', 'Dye concentration (mg/L)']
data, _ = prepare_data(inputs=inputs, outputs="k")

input_features = data.columns.tolist()[0:-1]

print(input_features)

# %%

output_features = data.columns.tolist()[-1:]
print(output_features)

# %%

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(
    data[input_features],
    data[output_features]
)

print(TrainX.shape, TestX.shape, TrainY.shape, TestY.shape)

# %%
# Evaluation with default parameters
# -------------------------------------

# %%
model = Model(
    model="DecisionTreeRegressor",
    input_features=input_features,
    output_features=output_features,
    verbosity=-1
)

model.fit(TrainX, TrainY.values)

# evaluate model performance
print(model.evaluate(
    TestX, TestY,
    metrics=["r2", "r2_score", "rmse"]))

# %%
# setup
# ----------

ITER = 0
VAL_SCORES = []
SUGGESTIONS = []
num_iterations = 100  # number of hyperparameter iterations
SEP = os.sep
PREFIX = f"hpo_{dateandtime_now()}"  # folder name where to save the results
algorithm = "tpe"
backend = "optuna"

# %%
# parameters space
# -----------------

param_space = [
    Categorical(["best", "random"], name='splitter'),
    Integer(low=2, high=10, name='min_samples_split'),
    Integer(low=1, high=40, name='max_depth'),
    #Integer(low=2, high=10, name="min_samples_leaf"),
    Real(low=0.0, high=0.005, name="min_weight_fraction_leaf"),
    #Categorical(categories=['sqrt', 'log2'], name="max_features"),
    #Integer(low=2, high=10, name="max_leaf_nodes"),
]

x0 = ['best',
      10,
    5,
      #5,
      0.1,
      #"sqrt",
    # #5
      ]

# %%
# objective function
# --------------------

def objective_fn(
        return_model:bool = False,
        **suggestions
)->Union[float, Model]:
    """
    The output of this function will be minimized
    :param return_model: whether to return the trained model or the validation
        score. This will be set to True, after we have optimized the hyperparameters
    :param suggestions: contains values of hyperparameters at each iteration
    :return: the scalar value which we want to minimize. If return_model is True
        then it returns the trained model
    """
    global ITER

    suggestions = jsonize(suggestions)
    SUGGESTIONS.append(suggestions)

    # build the model
    _model = Model(
        model={"DecisionTreeRegressor": suggestions},
        cross_validator={"KFold": {"n_splits": 10}},
        input_features=input_features,
        output_features=output_features,
        verbosity=-1
    )

    if return_model:
        _model.fit(TrainX.values, TrainY.values,
                  validation_data=(TestX, TestY.values))
        print(_model.evaluate(TestX, TestY,
                              metrics=["r2", "r2_score", "rmse"]))
        return _model

    # get the cross validation score which we will minimize
    val_score_ = _model.cross_val_score(TrainX.values, TrainY.values)[0]

    # since cross val score is r2_score, we need to subtract it from 1. Because
    # we are interested in increasing r2_score, and HyperOpt algorithm always
    # minizes the objective function
    val_score = 1 - val_score_

    VAL_SCORES.append(val_score)
    best_score = round(np.nanmin(VAL_SCORES).item(), 2)
    bst_iter = np.argmin(VAL_SCORES)

    ITER += 1

    print(f"{ITER} {round(val_score, 2)} {round(val_score_, 2)}. Best was {best_score} at {bst_iter}")

    return val_score

# %%
# running optimization loop
# -----------------------

optimizer = HyperOpt(
    algorithm=algorithm,
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=False,  # we can turn it False if we want post-processing of results
    opt_path=f"results{SEP}{PREFIX}",
    backend=backend,
)

# %%
res = optimizer.fit()

# %%
# postprocessing of results
# --------------------------
# print the best hyperparameters

print(optimizer.best_paras())

# %%
# convergence plot

optimizer.plot_convergence()
if SAVE:
    plt.savefig("results/figures/hpo_convergence.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

optimizer.plot_importance(with_optuna=True)
if SAVE:
    plt.savefig("results/figures/hpo_importance.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%
optimizer.plot_parallel_coords()
if SAVE:
    plt.savefig("results/figures/hpo_parallel_coords.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%
# Evaluation with optimized hyperparameters
# ------------------------------------------

bst_model = objective_fn(True, **optimizer.best_paras())
