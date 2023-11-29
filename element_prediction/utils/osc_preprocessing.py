
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from scipy.sparse.linalg import eigsh, eigs
import numpy as np


def center_scale(data, meanval = None, stdval = None):
    
    if meanval is None:
        meanval = np.nanmean(data)
    if stdval is None:
        stdval = np.nanstd(data)
        
    scaledval = (data-meanval)/stdval    
    
    return scaledval, meanval, stdval

class Simpls(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_components=5, 
                 center = False, stripped = False):
    
        self.n_components = n_components
        self.center = center
        self.stripped = stripped
        
    
    def fit(self,X,Y):
        xc = X.copy()
        yc = Y.copy()
        
        #
        ncomp = self.n_components
        nobj = xc.shape[0]
        npred = xc.shape[1]
        nresp = yc.shape[1]
        
        V, R = np.zeros((npred, ncomp)),np.zeros((npred, ncomp))
        TQ = np.zeros((ncomp,nresp))
        B = [np.zeros((npred,nresp))]*ncomp
        if not self.stripped:
            P = R.copy()
            U,TT = np.zeros((nobj, ncomp)),np.zeros((nobj, ncomp))
            fitted = [np.zeros((nobj,nresp))]*ncomp
        S = np.dot(xc.T,yc)
        
        for a in range(ncomp):
            if nresp == 1:
                qa = 1
            else:
                if nresp<npred:
                    qa = eigsh(np.dot(S.T,S))[1]
                    
            ra = np.dot(S, qa)
            ta = np.dot(xc, ra)
            if self.center:
                ta -= np.mean(ta)
            tnorm = np.squeeze(np.sqrt(np.dot(ta.T,ta)))
            ta = ta/tnorm
            ra = ra/tnorm
            pa = np.dot(xc.T,ta)
            qa = np.dot(yc.T,ta)
            va = pa
            if a > 0:
                va = va - np.dot(V, np.dot(V.T,pa))
            va = va/np.sqrt(np.dot(va.T,va))
            S = S - np.dot(va,np.dot(va.T,S))
            R[:,a] = np.squeeze(ra)
            TQ[a] = np.squeeze(qa)
            V[:,a] = np.squeeze(va)
            B[a] = np.dot(V[:,0:a], TQ[0:a])
            if not self.stripped:
                ua = np.dot(yc, qa)
                if a > 0:
                    ua = ua- np.dot(TT, np.dot(TT.T, ua))
                P[:,a] = np.squeeze(pa)
                TT[:,a] = np.squeeze(ta)
                U[:,a] = np.squeeze(ua)
                fitted[a] = np.dot(TT[:,0:a], TQ[0:a])
        
        self._coefficients = B
        self._tq = TQ
        self._v = V
        self._r = R        
        

class OSC(BaseEstimator, TransformerMixin):
    
    #https://github.com/leegs52/OPLSR/blob/main/utils/dosc.m
    #http://www.bdagroup.nl/content/Downloads/software/software.php
    def _dosc(self, X, Y, _seed = 0):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            _seed (int, optional): _description_. Defaults to 0.
        """
        xc = X.copy()
        
        self._xmean_center = np.mean(xc)
        #xc-=self._xmean_center
        
        #for i in range(self.nosc_components):
        # deflate x
        xinpinv = np.linalg.pinv(xc.T).T
        # project Y onto X #Ŷ=PxY (step 1)
        Yhat = np.dot(xc, np.dot(xinpinv, Y))
        #AŶX=X−PŶX (step 2)
        AyX = xc - np.dot(Yhat,np.dot(np.linalg.pinv(Yhat),xc))
        #AyX,_,_ = center_scale(AyX)
        #AyX/=np.linalg.norm(AyX)
        #colw = np.repeat(1/(AyX.shape[1]), AyX.shape[1]).reshape(-1,1)
        #roww = np.repeat(1/AyX.shape[0], AyX.shape[0]).reshape(-1,1)
        #normalize 
        #AyX = (AyX.T * np.sqrt(colw)).T * np.sqrt(roww)
        #X <- t(t(X) * sqrt(col.w)) * sqrt(row.w)
        # PCA
        covmatrix = np.dot(AyX, AyX.T)#/AyX.shape[0]
        
        np.random.seed(_seed)
        v0 = np.random.rand(min(covmatrix.shape))
        t_,mn =eigsh(covmatrix,k = self.nosc_components, v0=v0)
        #t_ 
        #t = pcaval.fit_transform(xc)[:,0]
        #pcaval= PCA()
        #t_ = pcaval.fit_transform(AyX)[:,0].reshape(-1,1)
        pinvX = np.linalg.pinv(xc.T,0.001).T
        
        r = np.dot(pinvX,mn) ## 

        #t=Xr
        t = np.dot(xc, r) # t=Xr
        # p=XTt(tTt)−1 (step 7)
        p = np.dot(xc.T, np.dot(
            t,np.linalg.inv(np.dot(t.T, t))))
        #X−XrpT
        xc = xc - np.dot(t, p.T)
        
        self.W_ortho_ = r
        self.P_ortho_ = p
        
        
    # this code is based on https://www.sciencedirect.com/science/article/pii/S0169743998001099
    # and its implementation on R https://rdrr.io/cran/mt/src/R/mt_osc.R
    def _osc_wold(self, X, Y):
        
        xc = X.copy()
        el = [] # residuals
        wl = [] # weigths
        pl = [] # loadings 
        tl = [] # scores
        
        #X_pinv, Y_pinv = _pinv2_old(X), _pinv2_old(Y)
        for i in range(self.nosc_components):
            pcaval = PCA()
            count = 0
            convval = 1
            t = pcaval.fit_transform(xc)[:,0]
            
            while convval > 0.001 and count < 20:
        
                invproduct = np.linalg.pinv(np.dot(Y.T,Y))
                # orthogonalized to y s 1yYY Y Y t
                t_ = t - np.dot(np.dot(np.dot(Y,invproduct),Y.T),t)
                #print(t_.shape)
                # getting weights to make Xw closer to t_
                if self.plsregression == 'pls':
                    plsmodel = PLSRegression(n_components=self.n_components,scale=False)    
                    plsmodel.fit(xc , t_.copy())
                    w = plsmodel.x_weights_[:,self.n_components-1] 
                if self.plsregression == 'simpls':
                    plsmodel = Simpls(n_components=self.n_components, center= False, stripped = False) 
                    plsmodel.fit(xc , t_.copy().reshape(-1,1))
                    w = np.squeeze(plsmodel._coefficients[self.n_components-1])
                    
                #
                #w = plsmodel.x_weights_[:,self.n_components-1]
                w = w / np.linalg.norm(w)  # normalized
                t_ = np.dot(xc,w)
                convval = np.sqrt(np.sum(np.square(t_-t))/np.sum(np.square(t_ )))
                t = np.squeeze(t_)#.reshape(-1,1)
                #print()
                count+=1

            t = t.reshape(-1,1) if len(t.shape) == 1 else t
            w = w.reshape(-1,1) if len(w.shape) == 1 else t
            
            p = np.dot(xc.T,np.dot(t,np.linalg.pinv(np.dot(t.T,t))))
            xc = xc- np.dot(t,p.T)
        
            el.append(xc)
            wl.append(w)
            pl.append(p)
            tl.append(t)
            
        self.W_ortho_ = np.hstack(wl)
        self.P_ortho_ = np.hstack(pl)
        self.T_ortho_ = np.hstack(tl)
    
    def __init__(self, n_components=5, n_osccomponents = 2, 
                 method = 'osc_wold', 
                 plsregression = 'simpls',
                 scalex = False, scaley = False):
        self.nosc_components = n_osccomponents
        self.n_components = n_components
        

        self.W_ortho_ = None
        self.P_ortho_ = None
        self.T_ortho_ = None
        self.method = method
        self.x_mean_ = None
        self.y_mean_ = None
        self.x_std_ = None
        self.y_std_ = None
        self.scalex = scalex
        self.scaley = scaley
        self.plsregression = plsregression
        
        
    def angle(self, Y):
        
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
            
        if self.y_mean_ is not None:
            newy,_,_ = center_scale(Y.copy(), self.y_mean_, self.y_std_)
        else:
            newy = Y.copy()
             
        nt = np.squeeze(self.T_ortho_)
        mvl = np.sqrt(np.sum(np.square(nt), axis = 0)*np.sum(np.square(newy)))

        angle = np.dot(nt.T, newy)

        norms = np.linalg.pinv(mvl.reshape(-1,1))
        ang = np.dot(angle.T, norms.T)

        return np.mean(np.arccos(ang) * 180 / np.pi)

    def fit(self, X, Y):
        
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
            
        if self.scalex:
            X, self.x_mean_,self.x_std_ = center_scale(X)
        if self.scaley:
            Y, self.y_mean_,self.y_std_ = center_scale(Y)
        
        if self.method == 'osc_wold':
            self._osc_wold(X, Y)    
        if self.method == 'dosc':
            self._dosc(X,Y)
        # getting score matrix
        
        
    def transform(self, X):
        
        if self.x_mean_ is not None:
            newx,_,_ = center_scale(X.copy(), self.x_mean_, self.x_std_)
        else:
            newx = X.copy()

        if self.method == 'osc_wold':
            x = newx - np.dot(newx, np.dot(self.W_ortho_,self.P_ortho_.T))
        if self.method == 'dosc':
            #xnewDOSC=xnew−rTxnewp
            #newx -= self._xmean_center 
            tnew = np.dot(newx,self.W_ortho_)
            
            x = newx - np.dot(tnew, self.P_ortho_.T)
            
        return x
        