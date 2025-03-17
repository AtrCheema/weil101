"""
============================
3. Feature Selection
============================
Now we know that which model works best for our problem. Now we will
perform feature selection using the best model.
"""

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from easy_mpl import bar_chart

from BorutaShap import BorutaShap

from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SequentialFeatureSelector

from utils import set_rcParams
from utils import version_info
from utils import prepare_data, LABEL_MAP, SAVE

# %%
for lib, ver in version_info().items():
    print(lib, ver)

# %%

set_rcParams()

TOP_K = 10

df, _ = prepare_data(outputs="k")

df = df.rename(columns=LABEL_MAP)

feature_names = df.columns.tolist()[0:-1]

X, y = df.iloc[:, 0:-1], df.iloc[:, -1].values

print(X.shape, y.shape)

# %%
# Information gain
# --------------------
importances = mutual_info_regression(
    X, y)

bar_chart(
    importances,
    feature_names,
    color="teal",
    sort=True,
    show=False,
)
plt.tight_layout()
plt.show()


# %%
# Chi-squared
# ------------

chi2_features = SelectKBest(f_regression, k=10)
X_kbest_features = chi2_features.fit_transform(
    X, y)

chi2_features = np.array(feature_names)[chi2_features.get_support()]
print(chi2_features)

chi2_features = chi2_features[0:TOP_K].tolist()

# %%
# Variance Threshold
# -------------------

v_threshold = VarianceThreshold(threshold=0)
v_threshold.fit(X)
v_threshold.get_support()

bar_chart(
    v_threshold.variances_,
    feature_names,
    color="teal",
    sort=True,
    show=False
)
plt.tight_layout()
plt.show()

vt_features = {k:v for k,v in zip(v_threshold.variances_, feature_names, )}
# sort_by_value
vt_features =  dict(sorted(vt_features.items(), key=lambda item: item[1], reverse=True))
vt_features = np.array(list(vt_features.values()))[0:TOP_K].tolist()

# %%
# Forward Feature Selection
# ---------------------------
# Starting with empty/minimal feature set and adding features one by one

rgr = DecisionTreeRegressor()
sfs_forward = SequentialFeatureSelector(
    rgr, n_features_to_select=TOP_K, direction="forward"
).fit(X, y)

ffs_features = np.array(feature_names)[sfs_forward.get_support()]
print(ffs_features)

ffs_features = ffs_features.tolist()

# %%
# Backward feature elimination
# -----------------------------
# Starting with a full set of features and removing one by one
# and everytime measuring the decrease in performance. Finally we rank the
# features, according to the decrease in performance they cause.

sfs_forward = SequentialFeatureSelector(
    rgr, n_features_to_select=TOP_K, direction="backward"
).fit(X, y)

bfe_features = np.array(feature_names)[sfs_forward.get_support()]
print(bfe_features)

bfe_features = bfe_features.tolist()

# %%
# Recursive Feeature Elimination
# -------------------------------
# It is similar to backward feature elemination.

rfe = RFE(DecisionTreeRegressor(), n_features_to_select=TOP_K,
          step=1)
rfe.fit(X, y)

# %%
rfe_features = np.array(feature_names)[rfe.get_support()]
print(rfe_features)

rfe_features = rfe_features[0:TOP_K].tolist()

# %%
# Tree based method
# ------------------

rgr = DecisionTreeRegressor().fit(X, y)
model = SelectFromModel(rgr, prefit=True)

tb_features = np.array(feature_names)[model.get_support()]
print(tb_features)

tb_features = tb_features[0:TOP_K].tolist()

# %%
# Boruta shap
# -------------
# The purpose of Boruta is to find a subset of features from all the given features,
# which are relevant for the given task. It creats a copy of a feature which is called
# shadow feature. Then the shadow feature is shuffled. The model is trained with the original
# feature plus the shuffled shadow feature. After that the feature importance of the original feature
# and shadow feature is calcualted using SHAP. If the SHAP importance of a shadow
# feature is more than the orignal feature, then it is rejected. The intuition is that,
# if a feature is important, then its shuffled version should not have more importnace
# than the original feature. Finally, Boruta shap method groups features, either
# as confirmed important, or confirmed rejected or tentative features. Since
# Boruta involves training the original model again and again, this can be
# extremely costly if the model training is time consuming.
# For theory see `this <https://www.jstatsoft.org/article/view/v036i11>`_ and
# `this <https://www.kaggle.com/code/residentmario/automated-feature-selection-with-boruta/notebook>`_ .

class MyBoruta(BorutaShap):
    def box_plot(self, data, X_rotation, X_size, y_scale, figsize):

        if y_scale=='log':
            minimum = data['value'].min()
            if minimum <= 0:
                data['value'] += abs(minimum) + 0.01

        order = data.groupby(by=["Methods"])["value"].mean().sort_values(ascending=False).index
        my_palette = self.create_mapping_of_features_to_attribute(
            maps= ['#B17BB2', '#EE9E9D', '#00ABAC',  '#B9E6FB'])
                 # 'yellow',   'red',     'green',    'blue'

        # Use a color palette
        plt.figure(figsize=(10, 7))
        ax = sns.boxplot(x=data["Methods"], y=data["value"],
                        order=order, palette=my_palette)

        if y_scale == 'log':ax.set(yscale="log")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=X_rotation, size=14)
        ax.tick_params(labelsize=14)
        ax.grid(visible=True, ls='--', color='lightgrey')
        ax.set_ylabel('Z-Score', fontsize=14)
        ax.set_xlabel('Features', fontsize=14,)
        plt.tight_layout()

        if SAVE:
            plt.savefig("results/figures/boruta_shap.png", dpi=600, bbox_inches="tight")
        return

# %%

model = DecisionTreeRegressor()

# %%

Feature_Selector = MyBoruta(model=model,
                              importance_measure='shap',
                              classification=False)

# %%
# We observed that the number of confirmed important and tentative
# features remained same after 50 ``n_trials``. At 50 ``n_trials``
# the total potential features were 12. Further increasing
# the ``n_trials`` only moved the features from 'tentative' category
# to 'confirmed important' until 400. For computational constraints on
# readthedocs, we are setting ``n_trials`` to 100.
# `z_score` on y-axis is a measure of importance and therefore, boxplots
# display the distribution of importance.

Feature_Selector.fit(
    X=X, y=y,
     n_trials=100,
     sample=False,
     train_or_test = 'test',
     normalize=True,
    verbose=True
)

# %%
# Boxplot of features. Features with grass green color
# are considered as confirmed important. The orange color represents
# confirmed rejected/unimportant.
Feature_Selector.plot()

# %%
if SAVE:
    Feature_Selector.results_to_csv('results/Boruta_results.csv')
# %%
# get the names selected features
print(Feature_Selector.Subset().columns)

br_features = Feature_Selector.Subset().columns[0:TOP_K].tolist()

# %%
# printing the common features among all methods

mi_features = {k:v for k,v in zip(feature_names, importances)}
# sort_by_value
mi_features =  dict(sorted(mi_features.items(), key=lambda item: item[1], reverse=True))
mi_features = np.array(list(mi_features.keys()))[0:TOP_K].tolist()

print(set(
    br_features + mi_features + chi2_features + vt_features +\
    ffs_features + bfe_features + rfe_features
))
