
from timeit import default_timer as timer
from datetime import timedelta
import math
import random
import numpy as np
import pandas as pd
import copy
import os
import tqdm

import xgboost as xgb

from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
from sklearn.utils.fixes import loguniform
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import SGDRegressor
from skmultiflow.core import BaseSKMObject, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin
from skmultiflow.utils import check_random_state

from ..utils.data_processing import split_dataintotwo, SplitIds, retrieve_datawithids


class RegressorChainM(BaseSKMObject, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin):
    """ Regressor Chains for multi-output learning.
    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=SGDRegressor)
        Each member of the ensemble is an instance of the base estimator.
    order : str (default=None)
        `None` to use default order, 'random' for random order.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    Notes
    -----
    Regressor Chains are a modification of Classifier Chains [1]_ for regression.
    References
    ----------
    .. [1] Read, Jesse, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank.
       "Classifier chains for multi-label classification."
       In Joint European Conference on Machine Learning and Knowledge Discovery in Databases,
       pp. 254-269. Springer, Berlin, Heidelberg, 2009.
    Examples
    --------

    """

    def __init__(self, base_estimator=SGDRegressor(), random_state=None):
        super().__init__()
        self.base_estimator = base_estimator
        #self.order = order
        self.random_state = random_state
        self.chain = None
        self.ensemble = None
        self.L = None
        self._random_state = None   # This is the actual random_state object used internally
        self.__configure()

    def __configure(self):
        self.ensemble = None
        self.L = -1
        self._random_state = check_random_state(self.random_state)

    def fit(self, X, y, sample_weight=None):
        """ Fit the model.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the target values of all samples in X.
        sample_weight: Not used (default=None)
        Returns
        -------
        self
        """
        N, self.L = y.shape
        L = self.L
        N, D = X.shape

        self.chain = np.arange(L)
        #if self.order is None:

        #if self.order == 'random':
        #    self._random_state.shuffle(self.chain)

        # Set the chain order
        y = y[:, self.chain]

        # Train
        self.ensemble = [copy.deepcopy(self.base_estimator) for _ in range(L)]
        XY = np.zeros((N, D + L - 1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0:L - 1]
        for j in range(self.L):
            self.ensemble[j].fit(XY[:, 0:D + j], y[:, j])
        return self

    def partial_fit(self, X, y, sample_weight=None):
        """ Partially (incrementally) fit the model.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.
        sample_weight: Not used (default=None)
        Returns
        -------
        self
        """
        if self.ensemble is None:
            # This is the first time that the model is fit
            self.fit(X, y)
            return self

        N, self.L = y.shape
        L = self.L
        N, D = X.shape

        # Set the chain order
        y = y[:, self.chain]

        XY = np.zeros((N, D + L - 1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0:L - 1]
        for j in range(L):
            self.ensemble[j].partial_fit(XY[:, 0:D + j], y[:, j])

        return self

    def predict(self, X):
        """ Predict target values for the passed data.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the target values for.
        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.
        """
        N, D = X.shape
        Y = np.zeros((N, self.L))

        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j - 1]])
            pred = self.ensemble[j].predict(X)
            Y[:, j] = np.squeeze(pred)

        # Unset the chain order (back to default)
        return Y[:, np.argsort(self.chain)]

    def reset(self):
        self.__configure()
        return self

    def predict_proba(self, X):
        """ Not implemented for this method.
        """
        raise NotImplementedError


    def validation(self,X, Y):

        """ Evaluate prediction values
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the target values for.
        Y : numpy.ndarray of shape (n_samples, n_targets)
            True values.
        Returns
        -------
        A pandas.Dataframe with all evaluation metrics.
        """
        Y_pred = self.predict(X)

        eval_metrics = []
        for i in range(Y_pred.shape[1]):

            eval_metrics.append(
                get_eval_metrics(Y[:, i], Y_pred[:, i]))
        
        return eval_metrics

    def _more_tags(self):
        return {'multioutput': True,
                'multioutput_only': True}



def set_model(type = 'pls',
              scaler = 'standardscaler', 
              param_grid = None, 
              cv = 5, 
              nworkers = -1):
    
    if scaler == 'minmax':
        scl = MinMaxScaler()
    if scaler == 'standardscaler':
        scl = StandardScaler()

    if type == 'pls':
        if param_grid is None:
            rdcomps = np.linspace(start = 1, stop = 50, num = 30)
            param_grid = [{'n_components':np.unique([int(i) for i in rdcomps])}]

        gs_pls = GridSearchCV( PLSRegression(),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)
    
        pipelinemodel = Pipeline([('scaler', scl), ('pls', gs_pls)])

    if type == 'svr_linear':
        if param_grid is None:
            param_grid = {'C': np.logspace(-2, 2, 10),
                          'gamma': np.logspace(-5, 1, 10)}

        ## model parameters
        gs_svm_linear  = GridSearchCV(SVR(kernel='linear'),
                                          param_grid,
                                          cv=3,
                                          n_jobs=nworkers)
        pipelinemodel = Pipeline([('scaler', scl), ('svr_linear', gs_svm_linear)])

    if type == 'svr_radial':
        if param_grid is None:
            #param_grid = {'C': loguniform.rvs(0.1, 1e2, size=20),
            #              'gamma': loguniform.rvs(0.0001, 1e-1, size=20)}
            param_grid = {'C': np.logspace(-2, 2, 20),
                          'gamma': np.logspace(-5, 1, 20)}
        ## model parameters
        gs_svm_radial  = GridSearchCV(SVR(kernel='rbf'),
                                          param_grid,
                                          cv=3,
                                          n_jobs=nworkers)

        pipelinemodel = Pipeline([('scaler', scl), 
        ('svr_radial', gs_svm_radial)])


    if type == 'xgb':
        if param_grid is None:
            param_grid = {
                    'min_child_weight': [1, 2, 4],
                    'gamma': [0.001,0.01,0.5, 1, 1.5, 2, 5],
                    'n_estimators': [100, 500],
                    'colsample_bytree': [0.3, 0.45],
                    'max_depth': [2,4,8,16,32],
                    'reg_alpha': [1.1, 1.2, 1.3],
                    'reg_lambda': [1.1, 1.2, 1.3],
                    'subsample': [0.7, 0.8, 0.9]
                    }

        xgbreg = xgb.XGBRegressor(
                        eval_metric="rmse",
                        random_state = 123
                )
        gs_xgb  = RandomizedSearchCV(xgbreg,
                               param_grid,
                               cv=cv,
                               n_jobs=nworkers,
                               n_iter = 50)

        pipelinemodel = Pipeline([('scaler', scl), ('xgb', gs_xgb)])

        
    if type == 'rf':
        param_grid = { 
            'n_estimators': [200],
            'max_features': [0.15, 0.3, 0.4],
            'max_depth' : [4,8,16],
            'min_samples_split' : [2,4],
            'max_samples': [0.7,0.9],
            #'max_leaf_nodes': [50, 100, 200],
            #'criterion' :['gini', 'entropy']
        }
        gs_rf = GridSearchCV( RandomForestRegressor(random_state = 42),
                                param_grid,
                                cv=3,
                                n_jobs=-1)
    
        pipelinemodel = Pipeline([('scaler', scl), ('rf', gs_rf)])

    
    if type == 'lasso':
        if param_grid is None:
            alphas = np.logspace(-5, -0.1, 20)
            param_grid = [{"alpha": alphas}]
        import warnings
        warnings.filterwarnings('ignore')    
        gs_lasso  = GridSearchCV(Lasso(random_state=42, max_iter=1000, tol = 0.001),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)
        
        pipelinemodel = Pipeline([('scaler', scl), ('lasso', gs_lasso)])
    
    if type == 'ridge':
        if param_grid is None:
            alphas = np.logspace(-5, -0.1, 20)
            param_grid = [{"alpha": alphas}]
            
        gs_lasso  = GridSearchCV(Ridge(random_state=42, max_iter=1000, tol = 0.001),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)
        
        pipelinemodel = Pipeline([('scaler', scl), ('ridge', gs_lasso)])
    

    return pipelinemodel



def eval_cv(ytrue, ypred):

    eval_metrics = []
    for cv in range(len(ytrue)):
        ytest = np.array(ytrue[cv])
        if len(ytest.shape)>1:
            ytest = np.ravel(ytest)

        eval_metrics.append(
            pd.DataFrame({
            'cv': [cv],
            'r2': r2_score(y_true=ytest,
                            y_pred=ypred[cv]),
            'rmse': math.sqrt(mean_squared_error(y_true=ytest,
                            y_pred=ypred[cv]))}))
    return pd.concat(eval_metrics)


def best_perform(listdf, metric = 'r2'):
    bestmetric = 0
    idbestmetric = -1
    for i in range(len(listdf)):
        metricvalue = listdf[i][metric].mean()
        if metricvalue > bestmetric:
            
            bestmetric = metricvalue
            idbestmetric = i
    
    return bestmetric, idbestmetric

def get_xyvaldata(X, Y, kfolds=None, kifold = None, split_ids = None, phase = 'training'):

    if split_ids is None:
        split_ids = SplitIds(X.shape[0])
    
    kifold = 0 if kifold is None else kifold
    
    if kfolds is None:
        
        if phase == "validation":
            tr_x, val_x = split_dataintotwo(X, 
                                        idsfirst = split_ids.training_ids, 
                                        idssecond = split_ids.test_ids)

            tr_y, val_y = split_dataintotwo(Y, 
                                        idsfirst = split_ids.training_ids, 
                                        idssecond = split_ids.test_ids)
        if phase == "training":
            tr_x = retrieve_datawithids(X, split_ids.training_ids) 
            tr_y = retrieve_datawithids(Y, split_ids.training_ids)
            val_x, val_y = None, None

    else:
        
        tr_x, val_x = split_dataintotwo(X, 
                                        idsfirst = split_ids.kfolds(kfolds)[kifold][0], 
                                        idssecond = split_ids.kfolds(kfolds)[kifold][1])

        tr_y, val_y = split_dataintotwo(Y, 
                                        idsfirst = split_ids.kfolds(kfolds)[kifold][0], 
                                        idssecond = split_ids.kfolds(kfolds)[kifold][1])

    return tr_x, tr_y, val_x, val_y

def mae(real, prediction):
    real, prediction = check_real_predictionshapes(real, prediction)
    real, prediction = np.array(real), np.array(prediction)
    return np.mean(np.abs(real - prediction))

def prmse(real, prediction):
    real, prediction = check_real_predictionshapes(real, prediction)
    EPSILON =  1e-10 
    return (np.sqrt(np.mean(np.square((real - prediction) / (real + EPSILON))))) * 100


def check_real_predictionshapes(real, prediction):
    if not len(real.shape) == len(prediction.shape):
        if real.shape > prediction.shape:
            real = np.squeeze(real)
        else:
            prediction = np.squeeze(prediction)
    
    return real, prediction



def get_eval_metrics(real, prediction):
    return (pd.DataFrame({
                'r2': [r2_score(y_true=real,
                                y_pred=prediction)],
                'rmse': [math.sqrt(mean_squared_error(y_true=real,
                                y_pred=prediction))],
                'prmse': [prmse(real=real,
                                prediction=prediction)],
                'mae': [mae(real=real,
                                prediction=prediction)]}))

def check_resultsinreporter(reporter, eoi, chain,typetolookfor = 'element'):
    
    valresult = -1
    if typetolookfor == 'chain':
        
        valresult = reporter.get_chain_results(chain=chain, 
                             acc_score=['r2','rmse','prmse','mae', 'cv']) 
        
    if typetolookfor == 'element':
        valresult = reporter.get_element_results(element = eoi)

    if valresult != -1:
        valresult = pd.DataFrame(valresult)
        valresult['id'] = eoi
        valresult['chain'] = chain

    return valresult
class ElementsChainRegressor(RegressorChainM):

    def _erc_performances(self,X, Y, kfolds, chain, verbose = True):
            valmetrics = []
            if verbose:
                loopval = tqdm.tqdm(range(kfolds))
            else:
                loopval = range(kfolds)
                
            for k in loopval:
                tr_x, tr_y, val_x, val_y = get_xyvaldata(X, Y, kfolds=kfolds, kifold = k, split_ids = self.split_ids)
        
                #initialmodel = copy.deepcopy(self.base_estimator)
                m = self.fit(tr_x.to_numpy(), tr_y.to_numpy())
                pdmetric = self.validation(val_x.to_numpy(), val_y.to_numpy())
                for i, noi in enumerate(chain.split('_')):
                    pdmetric[i]['id'] = noi
                pdmetric = pd.concat(pdmetric)
                pdmetric['cv'] = k
                pdmetric['chain'] = chain
                valmetrics.append(pdmetric)
                self.reset()
                del m

            return pd.concat(valmetrics)


    def cv_single_output_validation(self, Y_pred):

        eval_metrics = []
        noi = list(self.single_ouputmodels_trained.keys())[0]

        noidata = self.element_concentrations.data_elements([noi])
        for i in range(len(Y_pred)):
            _, _, _, val_y = get_xyvaldata(self.features, 
                                               noidata, 
                                               kfolds=len(Y_pred), 
                                               kifold = i, 
                                               split_ids = self.split_ids)
            pdmetric = get_eval_metrics(val_y, Y_pred[i])               
            pdmetric['cv'] = [i]
            eval_metrics.append(pdmetric)
        
        return eval_metrics
    
    def cv_single_output_prediction(self):

        #N, D = self.features.shape
        noi = list(self.single_ouputmodels_trained.keys())[0]
        nmodels = len(self.single_ouputmodels_trained[noi])
        noidata = self.element_concentrations.data_elements([noi])
        Y = []
        for m in range(nmodels):
            _, _, val_x, _ = get_xyvaldata(self.features, 
                                               noidata, 
                                               kfolds=nmodels, 
                                               kifold = m, 
                                               split_ids = self.split_ids)

            pred = self.single_ouputmodels_trained[noi][m].predict(val_x)
            Y.append(np.squeeze(pred))
        
        return Y


    def cv_sinle_output_fit(self, element_to_fit = None, kfolds = None):
        
        #if kfolds is None:
        single_output_models = [0]*kfolds
        elementsdata = self.element_concentrations.data_elements([element_to_fit])
        self.single_ouputmodels_trained = {}

        for k in range(kfolds):
            tr_x, tr_y, _, _ = get_xyvaldata(self.features, elementsdata, kfolds=kfolds, kifold = k, split_ids = self.split_ids)
            m = self.base_estimator.fit(tr_x.to_numpy(), tr_y.to_numpy().ravel())
            single_output_models[k] = copy.deepcopy(self.base_estimator)
            self.base_estimator = copy.deepcopy(self._base_estimator)
        
        self.single_ouputmodels_trained[element_to_fit] = single_output_models


    def fit_chain(self, chain, x = None, y = None, phase = "validation"):

        elementsdata = self.element_concentrations.data_elements(chain.split('_'))
        if x is None and y is None:
            
            tr_x, tr_y, _, _ = get_xyvaldata(self.features, elementsdata, phase=phase, split_ids = self.split_ids)
        else:
            tr_x, tr_y = x, y 
        print('training: {} elements {}'.format(tr_x.shape[0],elementsdata.columns))
        return self.fit(tr_x.to_numpy(), tr_y.to_numpy())
    

    def predict_using_chain(self, chain):
        elementsdata = self.element_concentrations.data_elements(chain.split('_'))
        
        elementsdata = self.element_concentrations.data_elements(chain.split('_'))
        _, _, val_x, val_y = get_xyvaldata(self.features, elementsdata, phase="validation", split_ids = self.split_ids)
        y_pred = self.predict(val_x.to_numpy())[:,len(chain.split('_'))-1]

        pass


    def find_best_chain(self, element_to_predict, 
                        kfolds = None, not_include = None, 
                        checkpoint_path = None, suffix_check = '', thresh = 0.4, chain = None,
                        reporter = None):
        """
        function to find the best chain for the multi-target regression. The function will repeat

        Args:
            element_to_predict (_type_): a terget element for prediction
            kfolds (_type_, optional): number of folds for the evaluation. Defaults to None.
            not_include (_type_, optional): _description_. Defaults to None.
            checkpoint_path (_type_, optional): _description_. Defaults to None.
            suffix_check (str, optional): _description_. Defaults to ''.
            thresh (float, optional): _description_. Defaults to 0.4.
            chain (_type_, optional): _description_. Defaults to None.
            reporter (_type_, optional): _description_. Defaults to None.
        """
        def erc_performances(X, Y, kfolds, chain):
            valmetrics = []
            for k in range(kfolds):
                tr_x, tr_y, val_x, val_y = get_xyvaldata(X, Y, kfolds=kfolds, kifold = k, split_ids = self.split_ids)
        
                #initialmodel = copy.deepcopy(self.base_estimator)
                m = self.fit(tr_x.to_numpy(), tr_y.to_numpy())
                pdmetric = self.validation(val_x.to_numpy(), val_y.to_numpy())
                for i, noi in enumerate(chain.split('_')):
                    pdmetric[i]['id'] = noi
                pdmetric = pd.concat(pdmetric)
                pdmetric['cv'] = k
                pdmetric['chain'] = chain
                valmetrics.append(pdmetric)
                self.reset()
                del m

            return pd.concat(valmetrics)

        if element_to_predict is None:
            element_to_predict = self.element_concentrations.data.columns[0]

        
        availablelements = copy.deepcopy(list(self.element_concentrations.data.columns))
        availablelements.remove(element_to_predict)

        underlimit = True
        if chain is not None:
            elementspre = chain.split('_')
            for ele in elementspre:
                if ele in availablelements:
                    availablelements.remove(ele)
            r2ref = 0
            j = len(elementspre)-1
        else:
            j = 0
            chain = element_to_predict
        
            
        ### use all availaible elements for prediction
        print('initial elements ', availablelements)
        while len(availablelements)>0 and underlimit:
            skip = False
            #chainelements = copy.deepcopy(self.element_concentrations.data.columns)
            if j == 0:
                if reporter:
                    val_singleelement = check_resultsinreporter(reporter, element_to_predict, chain,
                                                                typetolookfor = 'chain')
                    if type(val_singleelement) is pd.DataFrame:
                        r2ref = val_singleelement.r2.mean()
                        print('{}: reference accuracy: {:.3f}'.format(element_to_predict,r2ref))
                        #j +=1
                        skip = True
        
                #elementsdata = self.element_concentrations.data_elements([element_to_predict])
                if not skip:
                    self.cv_sinle_output_fit(element_to_fit=element_to_predict,kfolds=10)
                    pred = self.cv_single_output_prediction()
                    m = self.cv_single_output_validation(pred)
                    val_singleelement = pd.concat(m)
                    val_singleelement['id'] = element_to_predict
                    val_singleelement['chain'] = chain
                    #val_singleelement = training_initialmodel(self.features, elementsdata, kfolds)
                    r2ref = val_singleelement.r2.mean()
                    print('{}: reference accuracy: {:.3f}'.format(element_to_predict,r2ref))

                if checkpoint_path is not None:
                        outputpath = os.path.join(checkpoint_path, suffix_check)
                        #TODO function for saving checkpoint
                        if not os.path.exists(outputpath):
                            os.mkdir(outputpath)
                        val_singleelement.reset_index().to_csv(os.path.join(outputpath, '{}.csv'.format(element_to_predict)))

            else:
                if not_include is not None:
                    for i in not_include:
                        if i in availablelements:
                            availablelements.remove(i)
                        
                comparemodels = []
                for element in availablelements:
                    skip = False
                    newchain = "{}_{}".format(element, chain)
                    
                    if reporter:
                        noisubset = check_resultsinreporter(reporter, element_to_predict, 
                                                            newchain,typetolookfor = 'chain')
                        if type(noisubset) is pd.DataFrame:
                            print('chain {} perfomance: {:.3f}'.format(newchain, noisubset.r2.mean()))
                            skip = True
                    
                    if not skip:
                            
                        elementsdata = self.element_concentrations.data_elements(newchain.split('_'))
                        valmetrics = erc_performances(self.features, elementsdata, kfolds, newchain)
                        noisubset = valmetrics.loc[valmetrics.id == element_to_predict]
                        print('chain {} perfomance: {:.3f}'.format(newchain, noisubset.r2.mean()))
                        if reporter:
                            ## save reporter
                            results = {i:list(noisubset[i].values.astype(float)) 
                                    if noisubset[i].values.dtype == np.int64 
                                    else list(noisubset[i].values) for i in noisubset.columns}

                            reporter.update_reporter(results)
                            reporter.save_reporter(reporter.fn)
                            
                        if checkpoint_path is not None:
                            outputpath = os.path.join(checkpoint_path, suffix_check)
                            if not os.path.exists(outputpath):
                                os.mkdir(outputpath)
                            valmetrics.reset_index().to_csv(os.path.join(outputpath, '{}.csv'.format(newchain)))

                    comparemodels.append(noisubset)
                    #print(valmetrics.loc[:,valmetrics.id == element_to_predict])
                bestperform, idbest = best_perform(comparemodels, metric = 'r2')

                if bestperform > thresh and (bestperform - r2ref) > 0:
                    
                    newelement = availablelements[idbest]
                
                    availablelements.remove(newelement)

                    chain = "{}_{}".format(newelement, chain)
                    r2ref = bestperform
                    print('the chain was updated : {}'.format(chain))
                    print('remaining elements ', availablelements)

                else:
                    underlimit = False

            j +=1

        return chain, comparemodels[idbest]

    def __init__(self, spdata, elements_concentration,base_estimator=..., 
                random_state=None,ids_order = None):


        RegressorChainM().__init__(base_estimator, random_state)
        self.random_state = random_state

        self.features = spdata
        self.element_concentrations = elements_concentration
        self.split_ids = ids_order
        self.base_estimator = copy.deepcopy(base_estimator)
        self._base_estimator = copy.deepcopy(base_estimator)

    
