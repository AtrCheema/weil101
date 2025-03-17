"""
===============
4. Model
===============
This notebooks shows modeling with DecisionTreeRegressor
"""

import numpy as np
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.postprocessing import ProcessPredictions
from ai4water.utils.utils import TrainTestSplit

from sklearn.tree import plot_tree

from utils import version_info
from utils import prepare_data, set_rcParams, SAVE

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%
# Lightweight Model
# ===================
# Using the features selected by Bortua-shap method

inputs = ['Solution pH', 'Time (m)', 'Anions', 'Ni (At%)', 'HA (mg/L)',
          'loading (g)', 'Pore size (nm)', 'O (At%)',
          'Light intensity (watt)', 'Mo (At%)', 'Dye concentration (mg/L)']
data, encoders = prepare_data(#inputs=inputs,
                              outputs="k")

print(data.shape)

# %%

input_features = data.columns.tolist()[0:-1]

# %%

output_features = data.columns.tolist()[-1:]

# %%

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(
    data[input_features],
    data[output_features]
)

print(TrainX.shape, TrainY.shape, TestX.shape, TestY.shape)

# %%

model = Model(
    model = "DecisionTreeRegressor",
    input_features=input_features,
    output_features=output_features,
    val_fraction=0.0,
)

model.fit(TrainX, TrainY.values)

# %%

print(model.evaluate(
    x=TrainX, y=TrainY.values,
    metrics=["r2", "r2_score", "rmse", "mae"]))

# %%
train_prediction = model.predict(
    x=TrainX)

# %%
pp = ProcessPredictions('regression', 1, show=False, save=False)
pp.regression_plot(TrainY, train_prediction,
                   ridge_line_kws={'color': '#67a278', 'lw': 2.0},
                   marker_color='#61a86b',
                   hist=True,
                   hist_kws={'color': '#a7c6a5'},
                   )
if SAVE:
    plt.savefig("results/figures/train_regression.png", dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

_ = pp.residual_plot(
    TrainY, train_prediction,
    hist_kws={'color': '#a7c6a5'},
    plot_kws=dict(color='#61a86b', markerfacecolor='#61a86b')
)
if SAVE:
    plt.savefig("results/figures/train_residual.png", dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()
# %%

ax = pp.edf_plot(
    TrainY, train_prediction,
    color='#61a86b',
    marker=('--', '-')
)
if SAVE:
    plt.savefig("results/figures/train_edf.png", dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()
# %%

print(model.evaluate(
    x=TestX, y=TestY.values,
    metrics=["r2", "r2_score", "rmse"]))

# %%
test_prediction = model.predict(
    x=TestX)
# %%
pp.regression_plot(
    TestY, test_prediction,
    ridge_line_kws={'color': '#c65d5d', 'lw': 2.0},
                   marker_color='#c74a52',
                   hist=True,
                   hist_kws={'color': '#d5998b'},
                   )

if SAVE:
    plt.savefig("results/figures/test_regression.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()
# %%

_ = pp.residual_plot(
    TestY, test_prediction,
    hist_kws={'color': '#d5998b'},
    plot_kws=dict(color='#c65d5d', markerfacecolor='#c65d5d')
)
if SAVE:
    plt.savefig("results/figures/test_residual.png", dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()
# %%

ax = pp.edf_plot(
    TestY, test_prediction,
    color='#c65d5d',
    marker=('--', '-')
)
if SAVE:
    plt.savefig("results/figures/test_edf.png", dpi=400, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%
all_x = data[input_features]
all_y = model.predict(all_x, process_results=False)
ax = model.prediction_analysis(
    features= 'Time (m)',
    x=all_x, y=all_y,
    kind="box",
    show=False
)
if SAVE:
    plt.savefig("results/figures/pred_analysis.png", dpi=400, bbox_inches="tight")
ax.set_yticklabels(np.round(ax.get_yticks(), 3))
plt.tight_layout()
plt.show()

# %%

ax = model.prediction_analysis(
    features= ['Time (m)',  'loading (g)'],
    x=all_x, y=all_y,
    num_grid_points=(6, 6),
    border=True,
    show=False,
    annotate_kws=dict(fontsize=12)
)
plt.tight_layout()
plt.show()

# %%
# get the hyperparameters of the learned decision tree

print(model._model.get_depth())
print(model._model.get_n_leaves())

# %%
# plotting decision tree
plot_tree(model._model)

# %%
# Heavyweight Model
# ===================
# Using all the input features

data, encoders = prepare_data(outputs="k")

print(data.shape)

# %%

input_features = data.columns.tolist()[0:-1]

# %%

output_features = data.columns.tolist()[-1:]

# %%

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(
    data[input_features],
    data[output_features]
)

print(TrainX.shape, TrainY.shape, TestX.shape, TestY.shape)


# %%

model = Model(
    model = "DecisionTreeRegressor",
    input_features=input_features,
    output_features=output_features,
    val_fraction=0.0,
)

model.fit(TrainX, TrainY.values)

# %%

print(model.evaluate(
    x=TrainX, y=TrainY.values,
    metrics=["r2", "r2_score", "rmse", "mae"]))

# %%
train_prediction = model.predict(
    x=TrainX)

# %%

print(model.evaluate(
    x=TestX, y=TestY.values,
    metrics=["r2", "r2_score", "rmse", "mae"]))
