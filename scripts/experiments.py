"""
===================
2. Experiments
===================
The purpose of this notebook is to compare performance of different
ML models on our whole dataset.
"""

import matplotlib.pyplot as plt

from ai4water.utils.utils import TrainTestSplit
from ai4water.experiments import MLRegressionExperiments

from utils import version_info
from utils import prepare_data, set_rcParams, SAVE

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%
# We run all the experiments considering first order decay rate as target

data, _ = prepare_data(outputs="k")

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

experiment = MLRegressionExperiments(
    input_features=input_features,
    output_features=output_features,

    show=False
)

# %%

experiment.fit(
    x=TrainX,
    y=TrainY.values,
)

# %%

r2 = experiment.compare_errors(
    'r2', x=TestX, y=TestY, figsize=(8, 10),
       cutoff_type='greater',
       cutoff_val=0.1,
             colors=("#005066", "#B3331D")
                               )
if SAVE:
    plt.savefig("results/figures/exp_r2.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

print(r2)

# %%

r2_score = experiment.compare_errors(
    'r2_score', x=TestX, y=TestY, figsize=(8, 10),
    cutoff_type='greater',
    cutoff_val=0.1,
    colors=("#005066", "#B3331D")
)
if SAVE:
    plt.savefig("results/figures/exp_r2_score.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

print(r2_score)
# %%

r2_score = experiment.compare_errors(
    'r2_score', x=TestX, y=TestY, figsize=(8, 10),
    cutoff_type='greater',
    cutoff_val=0.5,
    colors=("#005066", "#B3331D")
)
plt.tight_layout()
plt.show()
# %%

print(r2_score)

# %%

figure = experiment.taylor_plot(
    data=data,
    figsize=(8, 8),
    include=r2_score.index.tolist(),
    leg_kws={'facecolor': 'white',
             'edgecolor': 'black', 'bbox_to_anchor':(1.0, 0.9),
             'fontsize': 10, 'labelspacing': 1.0, 'ncol': 1
            },
)
figure.axes[0].axis['left'].label.set_text('')
figure.axes[0].set_title('Training')
if SAVE:
    plt.savefig("results/figures/exp_taylor.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()
