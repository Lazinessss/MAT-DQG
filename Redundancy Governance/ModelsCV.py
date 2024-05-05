import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score


def my_loss_func(y_true, y_pred):
    value = np.sqrt(mean_squared_error(y_true, y_pred))
    # value = sum(abs((y_pred -y_true) / y_true)) / len(y_true)
    return value


def my_loss_func2(y_true, y_pred):
    value = np.sum(np.sqrt(np.array(y_true) - np.array(y_pred)))
    return value


def mlr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    # kf = KFold(n_splits=5, shuffle=True, random_state=10)
    tuned_parameters = {}
    lr = GridSearchCV(LinearRegression(), tuned_parameters, cv=kf, scoring=score)

    lr.fit(train_x, train_y)
    return lr


def pls_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    # tuned_parameters = {'n_components': [4, 5, 6]}
    tuned_parameters = {'n_components': [4]}
    pls = GridSearchCV(PLSRegression(), tuned_parameters, cv=kf, scoring=score)

    pls.fit(train_x, train_y)
    # print(pls.best_params_)
    return pls


def ridge_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'alpha': [1, 5, 10]}
    ridge = GridSearchCV(Ridge(), tuned_parameters, cv=kf, scoring=score)

    ridge.fit(train_x, train_y)
    # print(ridge.best_params_)
    return ridge


def krr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)

    # kf = KFold(n_splits=5, shuffle=True, random_state=10)
    # tuned_parameters = {'alpha': [0.031623], 'gamma': [0.14678]}

    # chalcogenides dataset
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    # tuned_parameters = {'alpha': [1e-3, 1e-4, 1e-5], 'gamma': [1e-4, 1e-5, 1e-6]}
    tuned_parameters = {'alpha': [1e-3], 'gamma': [1e-4]}
    krr = GridSearchCV(KernelRidge(kernel='rbf'), tuned_parameters, cv=kf, scoring=score)

    krr.fit(train_x, train_y)
    # print(krr.best_params_)
    return krr


def lasso_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)

    # # NASICON dataset
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'alpha': [0.01, 0.001]}
    lasso = GridSearchCV(Lasso(max_iter=100000, tol=0.01), tuned_parameters, cv=kf, scoring=score)

    # chalcogenides dataset
    '''
    kf = KFold(n_splits=10, shuffle=True)
    tuned_parameters = {'alpha': np.linspace(1e-12, 1e-8, 100)}
    lasso = GridSearchCV(Lasso(), tuned_parameters, cv=kf)
    '''

    # lasso = GridSearchCV(Lasso(), tuned_parameters, cv=kf, scoring=score)

    lasso.fit(train_x, train_y)
    # print(lasso.best_params_)
    return lasso


def svr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)

    # NASICON dataset
    # '''
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'gamma': [1e-04, 1e-03], 'C': [50, 100]}
    # '''

    # kf = KFold(n_splits=5, shuffle=True, random_state=10)

    # perovskites1
    '''
    tuned_parameters = {'gamma': [0.010, 0.015], 'C': [80, 100]}
    tuned_parameters = {'gamma': [0.015], 'C': [100]}
    '''

    # perovskites2
    '''
    tuned_parameters = {'gamma': [1e-4, 2e-4, 3e-4], 'C': [40, 45, 50]}
    tuned_parameters = {'gamma': [3e-4], 'C': [50]}
    '''

    # FCC-solute
    '''
    tuned_parameters = {'gamma': [1e-3, 0.2e-1, 0.4e-1], 'C': [2, 15, 35, 45, 50]}
    tuned_parameters = {'gamma': [2e-2], 'C': [50]}
    '''

    # chalcogenides
    '''
    tuned_parameters = {'gamma': [5e-2, 1e-03, 5e-3, 1e-04], 'C': [100, 500, 1000]}
    tuned_parameters = {'gamma': [1e-03], 'C': [100]}
    '''

    # HEAs
    '''
    tuned_parameters = {'gamma': [5e-1, 8e-1, 5e-2], 'C': [5e3, 1e4, 5e4]}
    tuned_parameters = {'gamma': [5e-2], 'C': [1e4]}
    '''

    # superalloys
    '''
    # tuned_parameters = {'gamma': [5e-3, 1e-2, 5e-2], 'C': [10, 50, 100, 200]}
    # tuned_parameters = {'gamma': [5e-2], 'C': [10, 50]}
    '''

    svr = GridSearchCV(SVR(kernel='rbf'), tuned_parameters, cv=kf, scoring=score)

    svr.fit(train_x, train_y)
    # print(svr.best_params_)
    return svr


def knn_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    tuned_parameters = {'n_neighbors': [2, 3, 4]}
    # tuned_parameters = {'n_neighbors': [3]}
    knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=kf, scoring=score)

    knn.fit(train_x, train_y)
    # print(knn.best_params_)
    return knn


def gpr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = [{"kernel": [RBF(l) for l in [0.1]]}]
    gpr = GridSearchCV(GaussianProcessRegressor(alpha=0.001), tuned_parameters, cv=kf, scoring=score)

    gpr.fit(train_x, train_y)
    # print(gpr.best_params_)
    return gpr


def rf_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    tuned_parameters = {'n_estimators': [10, 100, 500]}
    # tuned_parameters = {'bootstrap': [False], 'criterion': ['mse'], 'max_depth': [20], 'max_features': [0.5786679072752287],
    #                     'min_samples_leaf': [1], 'min_samples_split': [4], 'n_estimators': [392], 'oob_score': [False],
    #                     'warm_start': [True]}
    rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=kf, scoring=score)

    rf.fit(train_x, train_y)
    return rf


def xgbr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    tuned_parameters = {'colsample_bytree': [0.6, 0.7], 'subsample': [0.6, 0.7]}
    # xgbr = GridSearchCV(XGBRegressor(n_estimators=300, max_depth=4), tuned_parameters, cv=kf, scoring=score)

    # xgbr.fit(train_x, train_y)
    # print(xgbr.best_params_)
    # return xgbr
    return 0


def dt_predictor(train_x, train_y):
    tuned_parameters = {'max_depth': [5], 'min_samples_split': [8]}
    dt = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5)

    # dt = DecisionTreeRegressor(max_depth=7, random_state=10)  # DK_NCOR

    dt.fit(train_x, train_y)
    # print(dt.best_params_)
    return dt


def predictors(train_x, train_y, type='MLR'):
    if type == 'SVR':
        model = svr_predictor(train_x, train_y)
    elif type == 'RF':
        model = rf_predictor(train_x, train_y)
    elif type == 'LASSO':
        model = lasso_predictor(train_x, train_y)
    elif type == 'Ridge':
        model = ridge_predictor(train_x, train_y)
    elif type == 'GKRR':
        model = krr_predictor(train_x, train_y)
    elif type == 'KNN':
        model = knn_predictor(train_x, train_y)
    elif type == 'GPR':
        model = gpr_predictor(train_x, train_y)
    elif type == 'DT':
        model = dt_predictor(train_x, train_y)
    elif type == 'PLS':
        model = pls_predictor(train_x, train_y)
    elif type == 'XGBR':
        model = xgbr_predictor(train_x, train_y)
    else:
        model = mlr_predictor(train_x, train_y)
    return model
