"""
==============================
1. Exploratory Data Analysis
==============================
In this file we analyze the data using some basic statistics
and exploratory plots. The purpose is to get familiarize with
the data

synthesis time: hydrothermal reaction time (time taken to prepare the material)

band gap: property of material (how much energy is required to excite one electron from outermost shell)

volume : volume of wastewater

"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE

from easy_mpl import boxplot, pie
from easy_mpl.utils import create_subplots, despine_axes, map_array_to_cmap

from utils import CATEGORIES
from utils import set_rcParams
from utils import SAVE, version_info
from utils import read_data, LABEL_MAP, plot_correlation, prepare_data

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%

data = read_data(outputs=['k', 'Efficiency'])

# %%
# printing number of rows and number of columns in the data

print(data.shape)

# # %%
# # printing counts of missing values
# data.isna().sum()

# # %%

# data.describe()

# # %%
# print(len(data['Catalyst'].unique()))

# data['Catalyst'].unique()

# # %%
# colors, _ = map_array_to_cmap(np.arange(len(data['Catalyst'].unique())), cmap="tab20")
# pie(data['Catalyst'], colors=colors)

# # %%
# # Anions are representative of inorganic content in the xyz.

# print(len(data['Anions'].unique()))

# data['Anions'].unique()

# # %%
# colors, _ = map_array_to_cmap(np.arange(len(data['Anions'].unique())), cmap="tab20")
# pie(data['Anions'], colors=colors)

# # %%

# print(len(data['Dye'].unique()))

# data['Dye'].unique()

# # %%
# colors, _ = map_array_to_cmap(np.arange(len(data['Dye'].unique())), cmap="tab20")
# pie(data['Dye'], colors=colors)

# # %%
# # Overall distribution of all features

# data_num = data.drop(columns=['Catalyst', 'Anions', 'Dye'])

# rearrange_columns in data_num so that same categories lie close to each other


# data_num_ = pd.DataFrame()
# for cat, val in CATEGORIES.items():
#     for v in val:
#         if v in data_num:
#             data_num_[v] = data_num[v]
# data_num_['k'] = data_num['k']
# data_num_['Efficiency'] = data_num['Efficiency']
# data_num = data_num_

# boxplot(data_num,
#         labels=[LABEL_MAP.get(label, label) for label in data_num.columns],
#         share_axes=False,
#         flierprops=dict(ms=2.0),
#         medianprops={"color": "black"},
#         fill_color='#01B0B9',
#         patch_artist=True,
#         show=False,
#         figsize=(7, 6),
#         )
# plt.subplots_adjust(wspace=0.05)
# plt.tight_layout()
# plt.show()

# # %%
# # Distribution of a feautre given a specific dye i.e.
# # Dist(Feature)|Dye

# grps = data.groupby(by="Dye")

# f, axes = create_subplots(data_num.shape[1], figsize=(11, 8))

# for col, ax in zip(data_num.columns, axes.flat):

#     _, out = boxplot(
#     [grps.get_group('Indigo')[col].values, grps.get_group('Melachite Green')[col].values],
#         flierprops=dict(ms=2.0),
#         medianprops={"color": "black"},
#         fill_color=['#005066', '#B3331D'],
#         widths=0.7,
#         patch_artist=True,
#         ax=ax,
#         show=False
#     )
#     ax.set_xlabel(LABEL_MAP.get(col, col))
#     ax.set_xticks([])
# ax.legend([out["boxes"][0], out["boxes"][1]], ['Indigo', 'Melachite Green'],
#           loc=(-2.5, -1))
# plt.subplots_adjust(wspace=0.65, hspace=0.4)
# if SAVE:
#     plt.savefig("results/figures/boxplots.png", dpi=600, bbox_inches="tight")
# plt.tight_layout()
# plt.show()

# # %%
# # correlation of all input features with k

# data_num1 = data_num.rename(columns=LABEL_MAP)
# data_num1.pop('Efficiency')
# plot_correlation(data_num1, show=False)
# if SAVE:
#     plt.savefig("results/figures/corr_all_k.png", dpi=600, bbox_inches="tight")
# plt.tight_layout()
# plt.show()
# # %%
# # correlation of all input features with Efficiency

# data_num1 = data_num.rename(columns=LABEL_MAP)
# data_num1.pop('k')
# plot_correlation(data_num1)

# # %%
# # correlation of all input features with Efficiency and k

# data_num1 = data_num.rename(columns=LABEL_MAP)

# plot_correlation(data_num1, show=False)
# if SAVE:
#     plt.savefig("results/figures/corr_all_k_e.png", dpi=600, bbox_inches="tight")
# plt.tight_layout()
# plt.show()

# # %%
# # correlation of only input features which were termed as important
# # by Boruta method for k.

# cols = ['Solution pH', 'Cat. Loading (g/L)', 'O (At%)', 'Pore Size (nm)',
#         'HA (mg/L)', 'Mo (At%)', 'Light Int. (W)', #'Anions',
#         'Initial Conc. (mg/L)', 'Rxn Time (min)', 'Ni (At%)']

# data_num1 = data_num.rename(columns=LABEL_MAP)[cols + ["k"]]
# plot_correlation(data_num1, annot_kws={"fontsize": 12}, show=False)
# if SAVE:
#     plt.savefig("results/figures/corr_k.png", dpi=600, bbox_inches="tight")
# plt.tight_layout()
# plt.show()
# # %%
# # plotting only those where correlation is higher than 0.6
# plot_correlation(data_num1, threshold=0.6, split="pos")

# # %%
# # plotting only those where correlation is below 0.5
# plot_correlation(data_num1, threshold=-0.5, split="neg")

# # %%
# # correlation of input features which were termed as important
# # by Boruta method.



# data_num1 = data_num.rename(columns=LABEL_MAP)[cols + ["Efficiency"]]
# plot_correlation(data_num1, annot_kws={"fontsize": 12})

# # %%

# data_num1 = data_num.rename(columns=LABEL_MAP)[cols + ["k", "Efficiency"]]
# plot_correlation(data_num1, annot_kws={"fontsize": 12},
#                  show=False)
# if SAVE:
#     plt.savefig("results/figures/corr_selected.png", dpi=600, bbox_inches="tight")
# plt.tight_layout()
# plt.show()
# # %%

# mpl.rcParams.update(mpl.rcParamsDefault)

# data, encoders = prepare_data(outputs=["k", "Efficiency"])
# input_features = data.columns.tolist()

# tsne = TSNE(random_state=313)
# comp = tsne.fit_transform(data[input_features])

# def scatter_(first, second, label,
#              axs, fig):

#     c = data[col].values

#     pc = axs.scatter(first, second,
#                     c=c,
#                     s = 2,
#             cmap="Spectral",
#             )

#     if label in ["Catalyst", "Anions", "Dye"]:
#         cmap = mpl.cm.Spectral
#         bounds = list(set(c))
#         norm = mpl.colors.BoundaryNorm(bounds + [bounds[-1]+1],
#                                        cmap.N)

#         colorbar = fig.colorbar(
#             mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#             ticks = bounds,
#                  ax=axs, orientation='vertical')

#         if label == "Catalyst":
#             ticklabels = range(18)
#         elif label == "Anions":
#             ticklabels = ['N/A', 'Cl', 'SO4', 'CO3', 'HCO3', 'HPO4']
#         elif label == "Dye":
#             ticklabels = ["MG", "INDIGO"]
#         else:
#             ticklabels = encoders[label].classes_
#         colorbar.ax.set_yticklabels(ticklabels)

#     else:
#         colorbar = fig.colorbar(pc, ax=axs)
#         label = LABEL_MAP.get(label, label)
#         colorbar.set_label(label)

#     despine_axes(colorbar.ax)
#     return


# f, axes = create_subplots(29, sharex="all", sharey="all",
#                        figsize=(9, 9))

# for col, ax in zip(input_features, axes.flat):

#     scatter_(comp[:, 0], comp[:, 1],
#          label=col, axs=ax, fig=f)

# plt.tight_layout()
# plt.show()
