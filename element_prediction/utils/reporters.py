

import os
from sklearn.metrics import f1_score
import json
import numpy as np


class EvaluateSuffix(object):
    @staticmethod
    def _check_json_suffix(fn):
        #assert fn[0](fn)
        return fn.endswith('.json')
            
    def __init__(self, arg) -> None:
        self._arg = arg
        
    def __call__(self, *args):
        
        if len(args) == 1:
            fn = args[0]
        else:
            fn = args[1]

        out =  self._arg(fn) if self._check_json_suffix(fn) else None
        return out
    
@EvaluateSuffix
def loadjson(fn):
    if os.path.exists(fn):
        with open(fn, "rb") as fn:
            reporter = json.load(fn)
    else:
        reporter = None
    return reporter

class ClassificationReporter(object):
    
    def update_reporter(self, new_entry):    
        for k in list(self._reporter_keys):
            self.reporter[k].append(new_entry[k])        
    
    def load_reporter(self, fn):    
        reporter = loadjson(fn)
        if reporter is None:
            reporter = {}
            for keyname in self._reporter_keys:
                reporter.update({keyname: []})
        else:
            print('load')
        self.reporter = reporter
    
    def scores_summary(self, scorenames = 'cvscores'):
        return [np.mean(score) for score in self.reporter[scorenames]]
    
    def best_score(self, scorenames = 'cvscores'):
        
        orderedpos = np.argsort(self.scores_summary(scorenames))
        
        rout = {}
        for keyname in self._reporter_keys:
            rout[keyname] = self.reporter[keyname][orderedpos[-1]]
        
        return rout
    
    def save_reporter(self, fn):
        json_object = json.dumps(self.reporter, indent=4)
        with open(fn, "w") as outfile:
            outfile.write(json_object)
    
    def __init__(self, _reporter_keys = None) -> None:
        
        if _reporter_keys is None:
            self._reporter_keys = ['features','cvscores']
        else:
            self._reporter_keys = _reporter_keys
    
class regressionReporter(object):
    
    def update_reporter(self, new_entry):    
        for k in list(self._reporter_keys):
            self.reporter[k].append(new_entry[k])        
    
    def reset_reporter(self):
        reporter = {}
        for keyname in self._reporter_keys:
            reporter.update({keyname: []})
        
        self.reporter = reporter
            
    def load_reporter(self, fn):    
        reporter = loadjson(fn)
        if reporter is None:
            self.reset_reporter()
        else:
            print('load')
            self.reporter = reporter
    
    def scores_summary(self, scorenames = 'cvscores'):
        return [np.mean(score) for score in self.reporter[scorenames]]
    
    def best_score(self, scorenames = 'cvscores'):
        
        orderedpos = np.argsort(self.scores_summary(scorenames))
        
        rout = {}
        for keyname in self._reporter_keys:
            rout[keyname] = self.reporter[keyname][orderedpos[-1]]
        
        return rout
    
    def save_reporter(self, fn):
        json_object = json.dumps(self.reporter, indent=4)
        with open(fn, "w") as outfile:
            outfile.write(json_object)
    
    def __init__(self, _reporter_keys = None) -> None:
        
        if _reporter_keys is None:
            self._reporter_keys = ['features','cvscores']
        else:
            self._reporter_keys = _reporter_keys


class MT_ElementReporter(regressionReporter):
    
    @property    
    def elements(self):
        assert len(self.reporter[self._element_key]) > 0
        uniqueelement = []
        [uniqueelement.append(i[0]) for i in self.reporter[self._element_key ] if i[0] not in uniqueelement]
        return uniqueelement

    @property   
    def chains(self):
        assert len(self.reporter[self._chain_key]) > 0
        uniquechain = []
        [uniquechain.append(i[0]) for i in self.reporter[self._chain_key] if i[0] not in uniquechain]
        return uniquechain

    def get_chain_results(self, chain, acc_score = ['r2']):
        poschain = -1
        try:
            for i in range(len(self.chains)):
                if self.chains[i] == chain:
                    poschain = i
                    break
            
            if poschain !=-1:
                value = {
                    evalme : self.reporter[evalme][poschain] for evalme in acc_score}
            else:
                value = -1
            
        except:
            # if the reporter is empty there will be an error, so this is an exception to skeep that
            value = -1 
            
        return value
    
    def singletarget_results(self, element, id_element = 'id', acc_score = ['r2']):
        
        posele = -1
        
        if type(self.reporter[id_element][0]) is list:
            raise ValueError('this function is only for multi-target reporter')
        
        try:
            for i in range(len(np.unique(self.reporter[id_element]))):
                if self.reporter[id_element][i] == element:
                    posele = i
                    break
            
            if posele !=-1:
                value = {
                evalme : self.reporter[evalme][posele] for evalme in acc_score}
            else:
                value = {
                evalme : np.nan for evalme in acc_score}
            
        except:
            # if the reporter is empty there will be an error, so this is an exception to skeep that
            value = -1 
            
        return value
    
    def multitarget_results(self, element, acc_score = ['r2','rmse','prmse','mae', 'cv']):
        
        try:
            bestchain = self.best_chainperelement(element=element)
            value = self.get_chain_results(chain = bestchain, acc_score = acc_score)
            
        except:
            #if the reporter is empty there will be an error, so this is an exception to skeep that
            value = -1 
            bestchain = -1
            
        return bestchain, value

    def best_chainperelement(self, element, acc_score = 'r2'):
                
        elementchainr = {i:np.mean(self.get_chain_results(chain=i)[acc_score]) 
                        for i in self.chains if i.endswith(element)}
        
        bestchain = list(elementchainr.keys())[np.argsort(list(elementchainr.values()))[-1]]

        return bestchain
    
    def __init__(self, _reporter_keys=None, element_key = 'id', chain_key='chain') -> None:
        super().__init__(_reporter_keys)
        self._element_key = element_key
        self._chain_key = chain_key
    