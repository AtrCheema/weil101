"""
==============
utils
==============
This file contains utility functions which are used
in other files.
"""

import os
import sys
from typing import Union, List, Tuple, Callable

import shap
import scipy
import mapie
import crepes
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns

from easy_mpl import plot, scatter
from easy_mpl.utils import create_subplots

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ai4water.eda import EDA
from ai4water.preprocessing import DataSet
from ai4water.utils.utils import get_version_info

# %%

SAVE = False

# %%
# We have 18 types of catalysts. We can however group them
# into following 7 broad categories.
CATALYST_CATEGORIES = {
    'LTH': 'LTH',  # Layered triple hydroxide
    'LM1': 'LM',
    'LM1.5': 'LM',
    'LM2': 'LM',
    'no catalyst': 'Photolysis',
    'pure BFO': 'BFO',
    '0.5 wt% Pd-BFO': 'Pd-BFO',
    '1 wt% Pd-BFO': 'Pd-BFO',
    '2 wt% Pd-BFO': 'Pd-BFO',
    '3 wt% Pd-BFO': 'Pd-BFO',
    '1 wt% Ag-BFO': 'Ag-BFO',
    '2 wt% Ag-BFO': 'Ag-BFO',
    '3 wt% Ag-BFO': 'Ag-BFO',
    '4 wt% Ag-BFO': 'Ag-BFO',
    '0.25 wt% Pt-BFO': 'Pt-BFO',
    '0.5 wt% Pt-BFO': 'Pt-BFO',
    '1 wt% Pt-BFO': 'Pt-BFO',
    '2 wt% Pt-BFO': 'Pt-BFO'
}
# %%

CATEGORIES = {
    "Physicochemical Properties": ["Catalyst", "Pore size (nm)",
                        "Pore volume (cm3/g)",
                        'Energy Band gap (Eg) eV',
                        'Surface area (m2/g)'
                        ],
    "Atomic Composition": ["O (At%)", "Mo (At%)", "Ni (At%)",
                              "S (At%)", "C (At%)", "Fe (At%)", "Al (At%)",
                              "Bi", "Ag", "Pd", "Pt"],
    "Dye Properties": ['log_Kw', 'hydrogen_bonding_acceptor_count',
                  'hydrogen_bonding_donor_count',
                  'solubility (g/L)', 'molecular_wt (g/mol)', 'Dye', 'pka1', 'pka2',
                       ],
    "Experimental Conditions": ['Hydrothermal synthesis time (min)',
                     'volume (L)',
                     "loading (g)",
                     "Light intensity (watt)",
                     'Light source distance (cm)',
                     "Time (m)",
                     'Dye concentration (mg/L)',
                     "Solution pH",
                     'HA (mg/L)',
                     "Anions",
                                'Mass ratio (Catalyst/Dye)'
                     ],
}

# %%

LABEL_MAP = {
    'Hydrothermal synthesis time (min)': 'Synth. Time (min)',
    'Energy Band gap (Eg) eV': "Band Gap (eV)",
    'Light source distance (cm)': "Light Dist. (cm)",
    'Dye concentration (mg/L)': "Initial Conc. (mg/L)",
    'Surface area (m2/g)': "Surface Area (m2/g)",
    'Pore volume (cm3/g)': "Pore Vol. (cm3/g)",
    'Light intensity (watt)': 'Light Int. (W)',
    'Catalyst_loading_mg': 'Cat. loading',
    'hydrogen_bonding_acceptor_count': 'HB acceptor count',
    'hydrogen_bonding_donor_count': 'HB donor count',
    'molecular_wt (g/mol)': 'M.W. (g/mol)',
    'solubility (g/L)': 'Solubility (g/L)',
    'volume (L)': 'Volume (L)',
    "Time (m)": 'Rxn Time (min)',
    "Bi" : "Bi (At%)",
    "Ag": "Ag (At%)",
    "Pd": "Pd (At%)",
    "Pt": "Pt (At%)",
    "log_Kw": "log Kow",
    'loading (g)': 'Cat. Loading (g/L)',
    "Pore size (nm)": "Pore Size (nm)",
    'Dye': 'Dyes',
    'pka1': 'pka1',
    'pka2': 'pka2',
    'Mass ratio (Catalyst/Dye)': 'Mass Ratio'
}

# %%

def read_data(
        inputs:Union[str, List[str]]=None,
        outputs:Union[str, List[str]] = None
)->pd.DataFrame:

    default_inputs = [
        'Catalyst', 'Hydrothermal synthesis time (min)',
       'Energy Band gap (Eg) eV', 'C (At%)', 'O (At%)', 'Fe (At%)', 'Al (At%)',
       'Ni (At%)', 'Mo (At%)', 'S (At%)', 'Bi', 'Ag', 'Pd', 'Pt',
       'Surface area (m2/g)', 'Pore volume (cm3/g)', 'Pore size (nm)',
       'volume (L)',

        # consider one of loading or catalysing loadnig
        'loading (g)', #'Catalyst_loading_mg',
       'Light intensity (watt)', 'Light source distance (cm)', 'Time (m)',

       'Dye',

        # pollutant (dye) properties)
        'log_Kw', 'hydrogen_bonding_acceptor_count', 'hydrogen_bonding_donor_count',
        'solubility (g/L)', 'molecular_wt (g/mol)', 'pka1', 'pka2',

        # instead of Ci we consider Dye Concentration
        'Dye concentration (mg/L)', 'Solution pH', #'Ci',
        'HA (mg/L)',
       'Anions',

        #'Mass ratio (Catalyst/Dye)'
    ]

    fpath = os.path.join(os.getcwd(), "data", "230613_Photocatalysis_with_Zeeshan_data_CMKim_Updated.csv")
    df = pd.read_csv(fpath)

    # first order k following https://doi.org/10.1016/j.seppur.2019.116195
    k = np.log(df["Ci"] / df["Cf"]) / df["Time (m)"]
    df["k"] = k

    k_2nd = ((1 / df["Cf"]) - (1 / df["Ci"])) / df["Time (m)"]
    df["k_2nd"] = k_2nd

    # at Time 0, let k==0
    df.loc[df['Time (m)'] <= 0.0, "k"] = 0.0

    # when final concentration is very low, k is not calculable (will be inf)
    # therefore inserting very small value of k
    df.loc[df['Cf']==0.0, "k"] = 0.001

    #mass_ratio = (loading / volume )/dye_conc.

    # when no anions are present, represent them as N/A
    df.loc[df['Anions'].isin(['0', 'without Anion']), "Anions"] = "N/A"

    if inputs is None:
        inputs = default_inputs

    if outputs is None:
        outputs = ['Efficiency']
    else:
        if not isinstance(outputs, list):
           outputs = [outputs]

    return df[inputs + outputs]

# %%

def _ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    # function for OHE
    assert isinstance(col_name, str)

    # setting sparse to True will return a scipy.sparse.csr.csr_matrix
    # not a numpy array
    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    df.pop(col_name)

    return df, cols_added, encoder

# %%

def prepare_data(
        inputs = None,
        outputs=None,
        encoding="le",
)->Tuple[pd.DataFrame, dict]:

    if encoding is not None:
        assert encoding in ("le", "ohe")

    data = read_data(inputs, outputs)

    cat_encoder, dye_encoder, anion_encoder = None, None, None
    encoders = {}
    if encoding=="ohe":
        # applying One Hot Encoding
        if 'Catalyst' in data.columns:
            data, _, cat_encoder = _ohe_column(data, 'Catalyst')

        if 'Dye' in data.columns:
            data, _, dye_encoder = _ohe_column(data, 'Dye')
        data, _, anion_encoder = _ohe_column(data, 'Anions')
    elif encoding == "le":
        # applying Label Encoding

        if 'Catalyst' in data.columns:
            data, cat_encoder = le_column(data, 'Catalyst')

        if 'Dye' in data.columns:
            data, dye_encoder = le_column(data, 'Dye')
        data, anion_encoder = le_column(data, 'Anions')

    # make sure that efficiency is the last column
    if outputs is None:
        data['Efficiency'] = data.pop("Efficiency")
    elif isinstance(outputs, list):
        for out in outputs:
            data[out] = data.pop(out)
    else:
        assert len(outputs) == 1
        output = outputs[0]
        data[output] = data.pop(output)

    encoders['Catalyst'] = cat_encoder
    encoders['Dye'] = dye_encoder
    encoders['Anions'] = anion_encoder

    return data, encoders

# %%

def le_column(df:pd.DataFrame, col_name)->tuple:
    """label encode a column in dataframe"""
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return df, encoder

# %%

def set_rcParams(**kwargs):
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    _kwargs = {
        'axes.labelsize': '14',
        'axes.labelweight': 'bold',
        'xtick.labelsize': '12',
        'ytick.labelsize': '12',
        'font.weight': 'bold',
        'legend.title_fontsize': '12',
        'axes.titleweight': 'bold',
        'axes.titlesize': '14',
        #'font.family': "Times New Roman"

    }

    if sys.platform == "linux":

        _kwargs['font.family'] = 'serif'
        _kwargs['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    else:
        _kwargs['font.family'] = "Times New Roman"

    if kwargs:
        _kwargs.update(kwargs)

    for k,v in _kwargs.items():
        plt.rcParams[k] = v

    return

# %%

def get_dataset(encoding="le", seed=313):

    data, encoders = prepare_data(
        encoding=encoding)

    dataset = DataSet(data=data,
                      seed=seed,
                      split_random=True,
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      )
    return dataset, encoders

# %%

def plot_correlation(df, show=True, **kwargs):
    eda = EDA(data=df, show=False)

    ax = eda.correlation(figsize=(9, 9), square=True,
                         cbar_kws={"shrink": .72},
                         cmap="RdYlGn",
                         **kwargs
                         )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold', rotation=70)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
    if show:
        plt.tight_layout()
        plt.show()
    return

# %%

def plot_ci(
        prediction,
        lower,
        upper,
        coverage:float,
        num_points=None,
        title= None,
        axes:plt.axes = None,
        legned:bool = True,
):
    if num_points:
        prediction = prediction[0:num_points]
        lower = lower[0:num_points]
        upper = upper[0:num_points]

    if axes is None:
        f, axes = plt.subplots()

    axes.plot(prediction,
                color='forestgreen',
                label='Prediction')
    axes.fill_between(np.arange(len(prediction)),
                      lower,
                      upper,
                      color="forestgreen",
                      label=f"{int(coverage * 100)}% CI",
                      alpha=0.6
                      )
    if legned:
        axes.legend()
    if title:
        axes.set_title(title)

    return axes

# %%

def ale_1d(
        predictor:Callable,
        X:pd.DataFrame,
        feature:str,
        bins:int = 10
):
    """creates 1d ale for a continuous feature
    Copying code from alepython package
    """
    quantiles = np.unique(
        np.quantile(
            X[feature], np.linspace(0, 1, bins + 1), interpolation="lower"
        )
    )

    indices = np.clip(
        np.digitize(X[feature], quantiles, right=True) - 1, 0, None
    )

    predictions = []
    for offset in range(2):
        mod_X = X.copy()
        mod_X[feature] = quantiles[indices + offset]
        predictions.append(predictor(mod_X))
    # The individual effects.
    effects = predictions[1] - predictions[0]

    index_groupby = pd.DataFrame({"index": indices, "effects": effects}).groupby(
        "index"
    )

    mean_effects = index_groupby.mean().to_numpy().flatten()

    ale = np.array([0, *np.cumsum(mean_effects)])

    ale = (ale[1:] + ale[:-1]) / 2

    ale -= np.sum(ale * index_groupby.size() / X.shape[0])
    return ale, quantiles


def plot_ale(
        predictor,
        X:pd.DataFrame,
        feature:str,
        bins:int = 10,
        ax=None,
        show:bool = True,
        **kwargs
):
    ale, q = ale_1d(predictor, X, feature, bins=bins)
    q = (q[1:] + q[:-1]) / 2
    ax = plot(q, ale, ax=ax, show=False, **kwargs)

    ax.set_xlabel(LABEL_MAP.get(feature, feature))
    ax.grid(ls=":", color="lightgrey")
    if show:
        plt.show()
    return

# %%

def shap_scatter_plots(
        shap_values:np.ndarray,
        TrainX:pd.DataFrame,
        feature_name:str,
        encoders,
        save:bool = True,
):
    """
    It is expected that the columns in TrainX and shap_values have same order.
    :param shap_values:
    :param TrainX:
    :param feature_name:
    :param encoders:
    :param save:

    :return:
    """
    f, axes = create_subplots(TrainX.shape[1],
                              figsize=(12, 9))

    index = TrainX.columns.to_list().index(feature_name)

    for idx, (feature, ax) in enumerate(zip(TrainX.columns, axes.flat)):

        clr_f_is_cat = False
        if feature in ['Anions', 'Catalyst']:
            clr_f_is_cat = True

        if feature in ['Catalyst', 'Anions']:
            enc = encoders[feature]
            dec_feature = pd.Series(
                enc.inverse_transform(TrainX.loc[:, feature].values),
                                    name=feature)
            if feature == 'Catalyst':
                dec_feature_d = {k: CATALYST_CATEGORIES[k] for k in dec_feature.unique()}
                color_feature = dec_feature.map(dec_feature_d)
            else:
                color_feature = dec_feature

            # instead of showing the actual names, we still prefer to
            # label encode them because actual names takes very large
            # space in figure/axes
            color_feature = pd.Series(
                LabelEncoder().fit_transform(color_feature),
                                      name=feature)
        else:
            color_feature = TrainX.loc[:, feature]

        color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)


        ax = shap_scatter(
            shap_values[:, index],
            feature_data=TrainX.loc[:, feature_name].values,
            feature_name=LABEL_MAP.get(feature_name, feature_name),
            color_feature=color_feature,
            color_feature_is_categorical=clr_f_is_cat,
            show=False,
            alpha=0.5,
            ax=ax
        )
        ax.set_ylabel('')

    plt.tight_layout()

    if save:
        feature_name = feature_name.replace(' ', '')
        feature_name = feature_name.replace('/', '_')
        plt.savefig(f"results/figures/shap_interac_{feature_name}.png", dpi=600, bbox_inches="tight")

    plt.show()

    return

# %%

def shap_scatter(
        feature_shap_values:np.ndarray,
        feature_data:Union[pd.DataFrame, np.ndarray, pd.Series],
        color_feature:pd.Series=None,
        color_feature_is_categorical:bool = False,
        feature_name:str = '',
        show_hist:bool = True,
        palette_name = "tab10",
        s:int = 70,
        ax:plt.Axes = None,
        edgecolors='black',
        linewidth=0.8,
        alpha=0.8,
        show:bool = True,
        **scatter_kws,
):
    """

    :param feature_shap_values:
    :param feature_data:
    :param color_feature:
    :param color_feature_is_categorical:
        whether the color feautre is categorical or not. If categorical then the
        array ``color_feature`` is supposed to contain categorical (either string or numerical) values which
        are then mapped to the color and are used prepare the legend box.
    :param feature_name:
    :param show_hist:
    :param palette_name:
        only relevant if ``color_feature_is_categorical`` is True
    :param s:
    :param ax:
    :param edgecolors:
    :param linewidth:
    :param alpha:
    :param show:
    :param scatter_kws:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()

    if color_feature is None:
        c = None
    else:
        if color_feature_is_categorical:
            if isinstance(palette_name, (tuple, list)):
                assert len(palette_name) == len(color_feature.unique())
                rgb_values = palette_name
            else:
                rgb_values = sns.color_palette(palette_name, color_feature.unique().__len__())
            color_map = dict(zip(color_feature.unique(), rgb_values))
            c= color_feature.map(color_map)
        else:
            c = color_feature.values.reshape(-1,)

    _, pc = scatter(
        feature_data,
        feature_shap_values,
        c=c,
        s=s,
        marker="o",
        edgecolors=edgecolors,
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
        show=False,
        **scatter_kws
    )

    if color_feature is not None:
        feature_wrt_name = ' '.join(color_feature.name.split('_'))
        if color_feature_is_categorical:
            # add a legend
            handles = [Line2D([0], [0],
                              marker='o',
                              color='w',
                              markerfacecolor=v,
                              label=k, markersize=8) for k, v in color_map.items()]

            ax.legend(title=feature_wrt_name,
                  handles=handles, bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      title_fontsize=14
                      )
        else:
            fig = ax.get_figure()
            # increasing aspect will make the colorbar thin
            cbar = fig.colorbar(pc, ax=ax, aspect=20)
            cbar.ax.set_ylabel(feature_wrt_name,
                               rotation=90, labelpad=14)

            cbar.set_alpha(1)
            cbar.outline.set_visible(False)

    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"SHAP value for {feature_name}")
    ax.axhline(0, color='grey', linewidth=1.3, alpha=0.3, linestyle='--')

    if show_hist:
        if isinstance(feature_data, (pd.Series, pd.DataFrame)):
            feature_data = feature_data.values
        x = feature_data

        if len(x) >= 500:
            bin_edges = 50
        elif len(x) >= 200:
            bin_edges = 20
        elif len(x) >= 100:
            bin_edges = 10
        else:
            bin_edges = 5

        ax2 = ax.twinx()

        xlim = ax.get_xlim()

        ax2.hist(x.reshape(-1,), bin_edges,
                 range=(xlim[0], xlim[1]),
                 density=False, facecolor='#000000', alpha=0.1, zorder=-1)
        ax2.set_ylim(0, len(x))
        ax2.set_yticks([])

    if show:
        plt.show()

    return ax

# %%

def version_info()->dict:
    info = get_version_info()
    info['crepes'] = crepes.__version__
    info['mapie'] = mapie.__version__
    info['shap'] = shap.__version__
    info['scipy'] = scipy.__version__
    info['matplotlib'] = matplotlib.__version__
    return info

# %%

def make_classes(exp):
    colors = {'Experimental Conditions': '#ed9571',
              'Physicochemical Properties': '#faebd7',
              'Atomic Composition': '#8a5a45',
              'Dye Properties': '#F3D4C4'
              }

    classes = []
    colors_ = []
    for f in exp.feature_names:
        if f in CATEGORIES['Experimental Conditions']:
            classes.append('Experimental Conditions')
            colors_.append(colors['Experimental Conditions'])
        elif f in CATEGORIES['Physicochemical Properties']:
            classes.append('Physicochemical Properties')
            colors_.append(colors['Physicochemical Properties'])
        elif f in CATEGORIES['Atomic Composition']:
            classes.append('Atomic Composition')
            colors_.append(colors['Atomic Composition'])
        elif f in CATEGORIES['Dye Properties']:
            classes.append('Dye Properties')
            colors_.append(colors['Dye Properties'])


    return classes, colors, colors_
