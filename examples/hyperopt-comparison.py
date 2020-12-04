"""
==========================
Comparing to hyperopt
==========================

Thomas Bury, December 2020.

.. currentmodule:: skopt

Sequential model-based optimization uses a surrogate
model to model the expensive to evaluate function `func`.
Skopt has been reported as much slower than hyperopt.
Using lightgbm as surrogate model, skopt almost as fast as
hyperopt, and as accurate.
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from time import time
from hyperopt import hp, tpe, fmin, Trials
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from tqdm import tqdm
from skopt.learning import LGBMQuantileRegressor
from skopt import lgbrt_minimize


def cat_var(df, col_excl=None, return_cat=True):
    """
    Categorical encoding (as integer). Automatically detect the non-numerical columns,
    save the index and name of those columns, encode them as integer, save the direct and inverse mappers as
    dictionaries.
    Return the data-set with the encoded columns with a data type either int or pandas categorical.

    :param df: pd.DataFrame
        the dataset
    :param col_excl: list of str, default=None
        the list of columns names not being encoded (e.g. the ID column)
    :param return_cat: bool, default=True
        return encoded object columns as pandas categoricals or not.
    :return:
     df: pd.DataFrame
        the dataframe with encoded columns
     cat_var_df: pd.DataFrame
        the dataframe with the indices and names of the categorical columns
     inv_mapper: dict
        the dictionary to map integer --> category
     mapper: dict
        the dictionary to map category --> integer
    """

    if col_excl is None:
        non_num_cols = list(set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number]))))
    else:
        non_num_cols = list(
            set(list(df.columns)) - set(list(df.select_dtypes(include=[np.number]))) - set(col_excl))

    # cat_var_index = [i for i, x in enumerate(df[col_names].dtypes.tolist()) if isinstance(x, pd.CategoricalDtype)
    # or x == 'object'] cat_var_name = [x for i, x in enumerate(col_names) if i in cat_var_index]

    cat_var_index = [df.columns.get_loc(c) for c in non_num_cols if c in df]

    cat_var_df = pd.DataFrame({'cat_ind': cat_var_index,
                               'cat_name': non_num_cols})

    # avoid having datetime objects as keys in the mapping dic
    date_cols = [s for s in list(df) if "date" in s]
    df[date_cols] = df[date_cols].astype(str)

    cols_need_mapped = cat_var_df.cat_name.to_list()
    inv_mapper = {col: dict(enumerate(df[col].astype('category').cat.categories)) for col in df[cols_need_mapped]}
    mapper = {col: {v: k for k, v in inv_mapper[col].items()} for col in df[cols_need_mapped]}

    progress_bar = tqdm(cols_need_mapped)
    for c in progress_bar:
        progress_bar.set_description('Processing {0:<30}'.format(c))
        df[c] = df[c].map(mapper[c]).fillna(0).astype(int)
        # I could have use df[c].update(df[c].map(mapper[c])) while slower,
        # prevents values not included in an incomplete map from being changed to nans. But then I could have outputs
        # with mixed types in the case of different dtypes mapping (like str -> int).
        # This would eventually break any flow.
        # Map is faster than replace

    if return_cat:
        df[non_num_cols] = df[non_num_cols].astype('category')
    return df, cat_var_df, inv_mapper, mapper


# adult data set
# classification of income > (<) 50k
col_names = ['age', 'workclass', 'fnlwgt', 'education',
             'education-num', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain',
             'capital_loss', 'hours_per_week', 'native_country',
             'income_bracket']

# read and pre-process
df = pd.read_csv("adult.zip", names=col_names)
df['target'] = (df['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df.drop('income_bracket', axis=1, inplace=True)

# integer encoding works better than OHE, see lightgbm documentation
train_data, cat_var_df, inv_mapper, mapper = cat_var(df, col_excl=None, return_cat=False)
X_train = train_data[[c for c in train_data.columns if c != 'target']].values
y_train = train_data['target'].values
all_features = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain',
                'capital_loss', 'hours_per_week', 'native_country']

###############
# hyperopt HPO
###############
hp_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
    'subsample': hp.uniform('subsample', 0.5, 1.),
}

# model and fit params
params = dict(learning_rate=0.01,
              num_boost_round=300,
              num_leaves=255,
              verbose=-1,
              is_unbalance=True)
fit_params = dict(feature_name=all_features,
                  categorical_feature=cat_var_df.cat_name.tolist())


def objective(params):
    clf = LGBMClassifier(**params, is_unbalance=True, verbose=-1, silent=True)
    score = cross_val_score(clf,
                            X_train, y_train,
                            scoring='f1',
                            cv=StratifiedKFold(random_state=3),
                            fit_params=fit_params).mean()
    return 1 - score


trials = Trials()
best = fmin(fn=objective,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

##########################
# skopt BayesSearchCV HPO
##########################
hh_space = dict(
    learning_rate=Real(0.01, 0.3),
    min_child_weight=Real(0.1, 10),
    colsample_bytree=Real(0.5, 1.),
    subsample=Real(0.5, 1.),
)

clf = LGBMClassifier(is_unbalance=True, verbose=-1, silent=True)
start = time()
opt = BayesSearchCV(clf,
                    search_spaces=hh_space,
                    scoring='f1',
                    cv=StratifiedKFold(random_state=3),
                    fit_params=fit_params,
                    optimizer_kwargs={'base_estimator': LGBMQuantileRegressor(), 'acq_optimizer': "sampling"},
                    n_iter=50,
                    n_jobs=-1)
opt.fit(X_train, y_train)
skopt_bayes_runtime = time() - start
print("The BayesSearchCV with lightGBM running time: {0:3.2f}s".format(skopt_bayes_runtime))

###################
# skopt lgbrt HPO
###################
# the space has to be tuples like these
hh_space_gbrt = [Real(0.01, 0.3, 'uniform', name='learning_rate'),
                 Real(0.1, 10, 'uniform', name='min_child_weight'),
                 Real(0.5, 1., 'uniform', name='colsample_bytree'),
                 Real(0.5, 1., 'uniform', name='subsample')]


# Let's adapt the objective
def gbrt_objective(params):
    tmp_params = {}
    tmp_params['learning_rate'], tmp_params['min_child_weight'], \
    tmp_params['colsample_bytree'], tmp_params['subsample'], = params[0], params[1], params[2], params[3]
    clf = LGBMClassifier(**tmp_params, is_unbalance=True, verbose=-1, silent=True)
    score = cross_val_score(clf,
                            X_train, y_train,
                            scoring='f1',
                            cv=StratifiedKFold(random_state=3),
                            fit_params=fit_params).mean()
    return 1 - score


start = time()
sk_best = lgbrt_minimize(gbrt_objective,
                         hh_space_gbrt,
                         n_calls=50,
                         verbose=False,
                         n_jobs=-1)
skopt_gbrt_runtime = time() - start
print("The skopt lightgbrt running time: {0:3.2f}s".format(skopt_gbrt_runtime))

##########
# Scoring
##########
print('best HYPEROPT F1 score: {0:1.3f}'.format(1 - trials.best_trial['result']['loss']))
# hyperopt minimises 1-score.
print('best SKOPT BayesSearchCV F1 score: {0:1.3f}'.format(opt.best_score_))
print('best SKOPT LGBRT F1 score: {0:1.3f}'.format(1 - sk_best.fun))
