"""
======================
7. Conformal Analysis
======================
Conformal analysis is a distribution free uncertainty quantification method. Its
purpose is to test the robustness of the trained machine learning method. The standard
prodcedure is to dived the data into three sets (training, calibration nand test set).
The model is trained on training data. The calibration set is used to select the heuristic
and then this heuristic is applied using the test set. The final robustness (uncertainty) is
calculated on the test set which is not shown to the model at any stage before this.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

from crepes import ConformalRegressor
from crepes.fillings import sigma_knn, binning

from ai4water.utils.utils import TrainTestSplit

from easy_mpl import plot, bar_chart
from easy_mpl.utils import create_subplots

from sklearn.tree import DecisionTreeRegressor

from mapie.subsample import Subsample
from mapie.metrics import regression_coverage_score
from mapie.quantile_regression import MapieQuantileRegressor, MapieRegressor

from utils import SAVE, version_info
from utils import prepare_data, set_rcParams, plot_ci

# %%
set_rcParams()

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%
LABELS = {
    "jackknife_plus": "Jackknife +",
    "cv_plus": "CV +"
}
# %%
# Use the inputs selected by Boruta Shap method
inputs = ['Solution pH', 'Time (m)', 'Anions', 'Ni (At%)', 'HA (mg/L)',
          'loading (g)', 'Pore size (nm)', 'O (At%)',
          'Light intensity (watt)', 'Mo (At%)', 'Dye concentration (mg/L)']
data, _ = prepare_data(inputs=inputs, outputs="k")

input_features = data.columns.tolist()[0:-1]

# %%

output_features = data.columns.tolist()[-1:]

# %%

y = data[output_features].values.reshape(-1,)

y = np.array([(y[i]-y.min())/(y.max()-y.min()) for i in range(len(y))])

TrainX, X_test, TrainY, y_test = TrainTestSplit(seed=313).split_by_random(
    data[input_features], y)

X_prop_train, X_cal, y_prop_train, y_cal = TrainTestSplit(seed=313).split_by_random(
    TrainX,
    TrainY)

# %%
model = DecisionTreeRegressor(random_state=313)

model.fit(X_prop_train, y_prop_train)

y_hat_cal = model.predict(X_cal)

residuals_cal = y_cal - y_hat_cal

y_hat_test = model.predict(X_test)

lowers = {}
uppers = {}

# %%
# Standard conformal regressors
# -------------------------------

cr_std = ConformalRegressor()

# %%
# We will use the residuals from the calibration set to fit the
# conformal regressor.

cr_std.fit(residuals=residuals_cal)

# %%
# We may now obtain prediction intervals from the point predictions
# for the test set; here using a confidence level of 99%.

coverage = 0.95
intervals_std = cr_std.predict(y_hat=y_hat_test,
                           confidence=coverage)

lowers["Standard"] = intervals_std[:, 0]
uppers["Standard"] = intervals_std[:, 1]

# %%
# Normalized conformal regressors
# ---------------------------------

sigmas_cal_knn = sigma_knn(X=X_cal, residuals=residuals_cal)

cr_norm_knn = ConformalRegressor()

cr_norm_knn.fit(residuals=residuals_cal, sigmas=sigmas_cal_knn)

# %%
sigmas_test_knn = sigma_knn(X=X_cal,
                            residuals=residuals_cal,
                            X_test=X_test)

intervals_norm_knn = cr_norm_knn.predict(
    y_hat=y_hat_test,
    sigmas=sigmas_test_knn,
)

lowers["Normalized"] = intervals_norm_knn[:, 0]
uppers["Normalized"] = intervals_norm_knn[:, 1]

# %%
# Mondrian conformal regressors
# --------------------------------

bins_cal, bin_thresholds = binning(values=sigmas_cal_knn, bins=10)

cr_mond = ConformalRegressor()

cr_mond.fit(residuals=residuals_cal, bins=bins_cal)

# %%

bins_test = binning(values=sigmas_test_knn, bins=bin_thresholds)

intervals_mond = cr_mond.predict(
    y_hat=y_hat_test, bins=bins_test)

lowers["Mondrian"] = intervals_mond[:, 0]
uppers["Mondrian"] = intervals_mond[:, 1]

# %%
prediction_intervals = {
    "Std CR":intervals_std,
    "Norm CR knn":intervals_norm_knn,
    "Mond CR":intervals_mond,
}

# %%
coverages = []
mean_sizes = []
median_sizes = []

for name in prediction_intervals.keys():
    intervals = prediction_intervals[name]
    coverages.append(np.sum([1 if (y_test[i]>=intervals[i,0] and
                                   y_test[i]<=intervals[i,1]) else 0
                            for i in range(len(y_test))])/len(y_test))
    mean_sizes.append((intervals[:,1]-intervals[:,0]).mean())
    median_sizes.append(np.median((intervals[:,1]-intervals[:,0])))

pred_int_df = pd.DataFrame({"Coverage":coverages,
                            "Mean size":mean_sizes,
                            "Median size":median_sizes},
                           index=list(prediction_intervals.keys()))

pred_int_df.loc["Mean"] = [pred_int_df["Coverage"].mean(),
                           pred_int_df["Mean size"].mean(),
                           pred_int_df["Median size"].mean()]

print(pred_int_df)

# %%

interval_sizes = {}
for name in prediction_intervals.keys():
    interval_sizes[name] = prediction_intervals[name][:,1] \
    - prediction_intervals[name][:,0]

plt.figure(figsize=(8,8))
plt.ylabel("CDF")
plt.xlabel("Interval sizes")

colors = ["b","r","g","y","k","m","c","orange", "teal"]

for i, name in enumerate(interval_sizes.keys()):
    if "Std" in name:
        style = "dotted"
    else:
        style = "solid"
    plt.plot(np.sort(interval_sizes[name]),
             [i/len(interval_sizes[name])
              for i in range(1,len(interval_sizes[name])+1)],
             linestyle=style, c=colors[i], label=name)
plt.grid(visible=True, ls='--', color='lightgrey')
plt.legend()
plt.show()

# %%
f, axes = create_subplots(
    3,
    sharex="all",
)
for idx, (strategy, ax) in enumerate(zip(
        lowers.keys(), axes.flat)):

    plot_ci(
        prediction=y_hat_test,
        lower = lowers[strategy].reshape(-1,),
        upper = uppers[strategy].reshape(-1,),
        title=strategy,
        coverage=0.95,
        num_points=70,
        axes=ax,
        legned=False if idx<5 else True
    )

plt.show()

# %%
# Comparison of Conformal Analysis methods in MAPIE
# ---------------------------------------------------

rgr_quant = LGBMRegressor(random_state=313,
                    alpha=0.05,  # 95% confidence
                    objective="quantile")

model_quant = rgr_quant.fit(X_prop_train, y_prop_train)

STRATEGIES = {
    "Jackknife": dict(method="base", cv=-1),
    "Jackknife +": dict(method="plus", cv=-1),
    "Jackknife minmx": dict(method="minmax", cv=-1),
    "Cv": dict(method="base", cv=10),
    "CV +": dict(method="plus", cv=10),
    "CV minmax": dict(method="minmax", cv=10),
    "Jackknife + ab": dict(method="plus", cv=Subsample(n_resamplings=50)),
    "Jackknife_minmx ab": dict(
        method="minmax", cv=Subsample(n_resamplings=50)
    ),
    "Quantile": dict(
        cv="split", #alpha=0.05
    )
}

y_pred, y_pis = {}, {}
for strategy, params in STRATEGIES.items():
    print(f"running {strategy}")
    if strategy == "Quantile":
        mapie = MapieQuantileRegressor(model_quant, **params)
        mapie.fit(TrainX, TrainY, X_calib=X_cal, y_calib=y_cal,
                  random_state=313)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test,
                                                          alpha=0.05)
    else:
        mapie = MapieRegressor(model, verbose=1,
                               n_jobs=4, **params)
        mapie.fit(TrainX, TrainY)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)

# %%

f, axes = create_subplots(
    len(STRATEGIES),
    sharex="all",
    figsize=(9, 7)
)
for idx, (strategy, ax) in enumerate(zip(STRATEGIES.keys(), axes.flat)):

    plot_ci(
        prediction=y_pred[strategy],
        lower=y_pis[strategy][:, 0].reshape(-1,),
        upper=y_pis[strategy][:, 1].reshape(-1,),
        title=strategy,
        coverage=0.95,
        num_points=70,
        axes=ax,
        legned=False if idx<8 else True
    )

plt.show()

# %%
interval_sizes = {}

for strategy in STRATEGIES.keys():
    size = y_pis[strategy][:, 1].reshape(-1,) - y_pis[strategy][:, 0].reshape(-1,)
    interval_sizes[strategy] = size
    plot(
        size,
        label=strategy,
        ax_kws=dict(ylabel="Prediction Interval Width"),
        show=False
    )
plt.show()

# %%

plt.figure(figsize=(8,8))
plt.ylabel("CDF")
plt.xlabel("Interval sizes")

for i, name in enumerate(interval_sizes.keys()):
    if "jacknife" in name:
        style = "dotted"
    else:
        style = "solid"
    plt.plot(np.sort(interval_sizes[name]),
             [i/len(interval_sizes[name])
              for i in range(1,len(interval_sizes[name])+1)],
             linestyle=style, c=colors[i], label=name)
plt.grid(visible=True, ls='--', color='lightgrey')
plt.legend()
plt.show()

# %%
for strategy in STRATEGIES.keys():
    if strategy not in ["Quantile"]:
        plot(
            y_pis[strategy][:, 1].reshape(-1,) - y_pis[strategy][:, 0].reshape(-1,),
                   label=strategy,
            ax_kws=dict(ylabel="Prediction Interval Width"),
            show=False
    )
plt.show()

# %%
plt.figure(figsize=(8,8))
plt.ylabel("CDF")
plt.xlabel("Interval sizes")

for i, strategy in enumerate(interval_sizes.keys()):
    if strategy not in ["Quantile"]:
        if "jacknife" in strategy:
            style = "dotted"
        else:
            style = "solid"
        plt.plot(np.sort(interval_sizes[strategy]),
                 [i/len(interval_sizes[strategy])
                  for i in range(1,len(interval_sizes[strategy])+1)],
                 linestyle=style, c=colors[i], label=strategy)
plt.grid(visible=True, ls='--', color='lightgrey')
plt.legend()
plt.show()

# %%
coverage = pd.DataFrame([
    [
        regression_coverage_score(
            y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
        ),
        (
            y_pis[strategy][:, 1, 0] - y_pis[strategy][:, 0, 0]
        ).mean()
    ] for strategy in STRATEGIES
], index=STRATEGIES, columns=["Coverage", "Width average"]).round(2)

print(coverage.head(10))

# %%

fig, (ax, ax2) = plt.subplots(1, 2, sharey="all")

bar_chart(
    coverage.iloc[:, 0],
    ax=ax,
    ax_kws=dict(xlabel="Coverage"),
    color="#005066",
    show=False
)
bar_chart(
    coverage.iloc[:, 1],
    ax=ax2,
    color="#B3331D",
    ax_kws=dict(xlabel="Average Width"),
    show=False
)
plt.tight_layout()
plt.show()

# %%
f, ax = plt.subplots(figsize=(8,6))
ax.set_ylabel("CDF")
ax.set_xlabel("Interval sizes")

for i, strategy in enumerate(interval_sizes.keys()):
    if strategy not in ["Quantile", "Jackknife_minmx ab",
                        "Jackknife minmx", "Jackknife + ab", "CV minmax"]:
        if "jacknife" in strategy:
            style = "dotted"
        else:
            style = "solid"
        ax.plot(np.sort(interval_sizes[strategy]),
                 [i/len(interval_sizes[strategy])
                  for i in range(1,len(interval_sizes[strategy])+1)],
                 linestyle=style, c=colors[i], label=strategy)

xlim = ax.get_xlim()
ax.set_xlim([xlim[0], 0.5])
ax.grid(visible=True, ls='--', color='lightgrey')
ax.legend()
if SAVE:
    plt.savefig("results/figures/conformal_ci.png", dpi=600, bbox_inches="tight")
plt.show()

# %%

strategies = ["Jackknife", "Jackknife +", "Cv", "CV +"]
f, axes = create_subplots(
    len(strategies),
    sharex="all",
    figsize=(7, 6)
)
for idx, (strategy, ax) in enumerate(zip(strategies, axes.flat)):

    plot_ci(
        prediction=y_pred[strategy],
        lower=y_pis[strategy][:, 0].reshape(-1,),
        upper=y_pis[strategy][:, 1].reshape(-1,),
        title=strategy,
        coverage=0.95,
        num_points=70,
        axes=ax,
        legned=False if idx<8 else True
    )

    if idx in [0, 2]:
        ax.set_ylabel("Normalized k")
    ax.set_xlabel("Samples")

plt.tight_layout()
if SAVE:
    plt.savefig("results/figures/conformal_interval_size.png", dpi=600, bbox_inches="tight")
plt.show()
