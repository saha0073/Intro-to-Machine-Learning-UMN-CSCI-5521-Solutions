def q3(): 
    #from __future__ import print_function
    import IPython
    #print('IPython:', IPython.__version__)
    import numpy  
    #print('numpy:', numpy.__version__)
    import numpy 
    import numpy as np
    import scipy
    #print('scipy:', scipy.__version__)
    import matplotlib
    #print('matplotlib:', matplotlib.__version__)
    import sklearn
    #print('scikit-learn:', sklearn.__version__)
    
    #from sklearn.model_selection import train_test_split
    #from sklearn.model_selection import cross_val_score
    #import scipy.io as sio
    # initialize a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    #from sklearn.metrics import accuracy_score
    
    ###############################################
    #data1 = sio.loadmat('data1.mat')
    #Xsf=data1['X']
    #ysf=data1['y']
    #Nrowy=len(ysf)
    #################################################
    ######################### Normalizing the X dataset
    def normalize(X):
        Nrown=len(X[:,0])
        Ncoln=len(X[1,:])
        m=np.zeros(Ncoln)
        s=np.zeros(Ncoln)
        Xnor=np.zeros(shape=(Nrown,Ncoln))
        for j in range(0,Ncoln):
            m[j]=np.mean(X[:,j])
            s[j]=np.std(X[:,j])
            #print(m[j],s[j])
            if (s[j]==0):
                #print(X[:,j])
                Xnor[:,j]=X[:,j]-m[j]
            else:
                Xnor[:,j]=(X[:,j]-m[j])/s[j]
        return Xnor
    ######################################
    
    from sklearn.datasets import load_boston
    boston = load_boston()
    Xbos=boston.data
    Xbos=normalize(Xbos)               #normalizing the X dataset (MySVM2 converges better)
    ybos=boston.target
    medy=np.median(ybos)
    y75th=np.percentile(ybos, 75)
    Nbos=len(ybos)
    #print(Nbos), print(medy), print(y75th)
    yb50=np.zeros(Nbos, dtype=int)
    #yp75=yp50
    for i in range(0, Nbos):
        ychk=ybos[i]
        if (ychk>=medy):
            yb50[i]=1
        #print(y[i])
        else:
            yb50[i]=0
    #print(yb50)
    #print(y)
    yb75=np.zeros(Nbos, dtype=int)      
    for i in range(0, Nbos):
        ychk1=ybos[i]
        if (ychk1>=y75th):
            yb75[i]=1
        #print(y[i])
        else:
            yb75[i]=0
            
    #print(yb75)
    #print(ybos)
    from sklearn.datasets import load_digits
    digits = load_digits()
    Xdig=digits.data
    Xdig=normalize(Xdig)                   #normalizing X dataset (myLogisticReg2 converges faster)
    ydig=digits.target
    
    
    
    
    #Xme=normalize(Xim)
    #yme=yim
    
    diag=False
    
    #my_cross_val(MyLogisticReg2,Xdig,ydig,5,False)
    #my_cross_val(MultiGaussClassify, Xdig, ydig,5, False)
    
    print('1. Error rates for MySVM2 with m=40 for Boston50')
    my_cross_val(MySVM2, Xbos, yb50,5, 40)
    print('\n','2.Error rates for MySVM2 with m=200 for Boston50')
    my_cross_val(MySVM2, Xbos, yb50,5,200)
    print('\n','3.Error rates for MySVM2 with m=n for Boston50')
    my_cross_val(MySVM2, Xbos, yb50,5, len(Xbos[:,0])*2)
    print('\n','4.Error rates for LogisticRegression with Boston50')
    my_cross_val(LogisticRegression, Xbos, yb50,5, len(Xbos[:,0])*2)
    print('\n','5. Error rates for MySVM2 with m=40 for Boston75')
    my_cross_val(MySVM2, Xbos, yb75,5, 40)
    print('\n','6.Error rates for MySVM2 with m=200 for Boston75')
    my_cross_val(MySVM2, Xbos, yb75,5,200)
    print('\n','7.Error rates for MySVM2 with m=n for Boston75')
    my_cross_val(MySVM2, Xbos, yb75,5, len(Xbos[:,0])*2)
    print('\n','8.Error rates for LogisticRegression with Boston75')
    my_cross_val(LogisticRegression, Xbos, yb75,5, len(Xbos[:,0])*2)
    
    return



class MySVM2():
    def __init__(self,mb):
        self.mb=mb
        #print(diag)
        #pass
    ######################################################    
    def sigmoid(w,X,ncls):
        import numpy as np
        #print('within sigmoid')
        #chk=self.chk
        #print('printing chk', chk)
        sum1=np.zeros(ncls)
        yret=np.zeros(ncls)
        #print(type(X))
        Nscol=len(X)
        #print(X)
        ysum=0
        for z in range(0,ncls):
            for j in range(0,Nscol):
                sum1[z]=sum1[z]+w[z,j]*X[j]
                #print(sum1[z])
            ysum=ysum+np.exp(sum1[z])
        #print(w,sum1,X)
        for z in range(0,ncls):
            yret[z]=np.exp(sum1[z])/ysum
        #y=1/(1+np.exp(-sum1))
        #print(y)
        return yret
    
    def multi(w,w0,X):
        import numpy as np
        #print('within sigmoid')
        #chk=self.chk
        #print('printing chk', chk)
        sum1=0
        #yret=np.zeros(ncls)
        #print(type(X))
        Nscol=len(X)
        #print(X)
        ysum=0

        for j in range(0,Nscol):
            sum1=sum1+w[j]*X[j]
        sum1=sum1+w0
        return sum1
    ######################################################
    def pred_class(w,X,ncls,cls):
        #P=np.zeros(ncls)+10
        ypred=MySVM2.multi(w,X,ncls)
        #for z in range(0,ncls):
            #P[z]=sigmoid(w[z],X)
        ymax=max(ypred)
        for z in range (0,ncls):
            if (ypred[z]==ymax):
                ychk=cls[z]
        return ychk
    ###############################################################
        
    def fit(self, X,y):
        import random
        import matplotlib.pyplot as plt
        mb=self.mb                    #size of mini batch
        lam=5         #regularization constant
        
        import numpy  
        import math
        #print('numpy:', numpy.__version__)
        import numpy as np
        from random import uniform
        from math import pow
        chk=25
        self.chk=chk
        #Xim=Xbos
        #yim=yb50
    
        ######################### Normalizing the X dataset
        def normalize(X):
            Nrown=len(X[:,0])
            Ncoln=len(X[1,:])
            m=np.zeros(Ncoln)
            s=np.zeros(Ncoln)
            Xnor=np.zeros(shape=(Nrown,Ncoln))
            for j in range(0,Ncoln):
                m[j]=np.mean(X[:,j])
                s[j]=np.std(X[:,j])
                #print(m[j],s[j])
                if (s[j]==0):
                    #print(X[:,j])
                    Xnor[:,j]=X[:,j]-m[j]
                else:
                    Xnor[:,j]=(X[:,j]-m[j])/s[j]
            return Xnor
        ######################################
        
        #Xtrain=normalize(X)
        Xtrain=X
        ytrain=y
        
        Nrow=len(ytrain)
        Ncol=len(Xtrain[1,:])
        
        
        ############################### Finding the # of classes
        def num_cls(y):
            clsn=np.array([y[0]])
            Nrown=len(y)
            for i in range (1,Nrown):
                #print(i)
                nclsn=np.size(clsn)
                count=0
                for j in range (0,nclsn):
                    if (y[i]!=clsn[j]):
                        count=count+1
                if (count==nclsn):
                    clsn=np.append(clsn,y[i])
            nclsn=np.size(clsn)   #total number of class
            #print(nclsn)
            #print(clsn)
            return nclsn,clsn
        ######################################################    
        #ncls,cls=num_cls(ytrain)
        #print(ncls,cls)
        
       
        #dw=np.zeros(shape=(ncls,Ncol))
        #Xld=np.zeros(shape=(Nrow,Ncol+1))
        '''
        r=np.zeros(shape=(ncls,Nrow), dtype=int)+50
        #r=[10]*Ntest 
        Pc=np.zeros(ncls)
        for z in range (0,ncls):
            for i in range (0,Nrow):
                if (ytrain[i]==cls[z]):
                    r[z,i]=1.0
                    Pc[z]=Pc[z]+1
                else:
                    r[z,i]=0.0
            Pc[z]=Pc[z]/Nrow
        #print(*r)
        '''
        ########################################## defining class labels as +1 and -1    
        ncls,cls=num_cls(ytrain)               
        r=np.zeros(shape=Nrow,dtype=int)
        for i in range(0,Nrow):
            if (ytrain[i]==cls[0]):
                r[i]=1
            else:
                r[i]=-1
        
        #ncls=1      #please change this, if required
        w=np.zeros(shape=Ncol)
        w0=uniform(-0.01,0.01)
        #Xld=Xme
        #yld=yme
        for j in range(0,Ncol):
            w[j]=uniform(-0.01,0.01)
        #for z in range(0,ncls):
            
        #print(w)
        #w=numpy.array([1,-1])
        dw=np.zeros(shape=Ncol)
        dw0=0
        et=0.01  #step size, please modify it if required
        #sum2=0
        num=0
        wdiff=100
        
        
        
        allin=np.zeros(shape=(Nrow),dtype=int)
        for i in range(0,Nrow):    #creating an array of indices
            allin[i]=i
        #narrin=random.shuffle(allin)
        wdar=np.zeros(shape=1)
        #wdar=[]
        while (wdiff>=.0000001 and num<=100):
            #for z in range(0,ncls):
                #for j in range(0,Ncol):
                    #dw[z,j]=0
            random.shuffle(allin)
            dw=0*dw
            dw0=0*dw0
            
            for t in range(0, mb):
                if (t<Nrow-10):             # while using full batch this condition will be required
                    i=allin[t]
                    #print(i)
                    yper=MySVM2.multi(w,w0,Xtrain[i])
                    #yper=np.matmul(w,Xtrain[i])
                    #print(yper)
                    #for z in range(0,ncls):
                    for j in range(0,Ncol):
                        if (r[i]*yper>=1):
                            dw[j]=dw[j]
                            dw0=dw0
                                #print('greater')
                        else:
                            dw[j]=dw[j]+r[i]*Xtrain[i,j]
                            dw0=dw0+r[i]
                                #print('smaller')
                            #dw[z,j]=dw[z,j]+(r[z,i]-yper[z])*Xtrain[i,j]
                            #dw[z,j]=(r[z,i]-yper[z])*Xtrain[i,j]
                            #if((math.isnan(yper[z]==True))):
                                #print(z,i,j)
            #print(*dw)
            wold=w
            w0old=w0
            wdiff=0
            #print('hi')
            dw=(dw-lam*w)/mb      #lambda term is added for regularization 
            #dw=dw/mb-lam*w
            dw0=dw0/mb
            w=wold+et*dw
            w0=w0old+et*dw0
            #print(w)
            #print(w0)
            #for z in range(0,ncls):
            for j in range(0,Ncol):
                #w[z,j]=wold[z,j]+float(0.1*dw[z,j])
                wdiff=wdiff+(w[j]-wold[j])*(w[j]-wold[j])
                #print(et*dw[z,j],w[z,j],wold[z,j])
            wdiff=wdiff+(w0-w0old)*(w0-w0old)
            wdiff=wdiff/(Ncol) ################ Here I am dividing wdiff by (ncls*Ncol)       
            num=num+1
            #print(num,wdiff)
            wdar=np.append([wdar],[np.log(wdiff)])
            #print(wold)
            #print(dw)
        #print(w)
        #self.W=W
        self.w=w
        self.w0=w0
        self.ncls=ncls
        self.cls=cls
        #print(w,ncls,cls)
        numa=np.zeros(shape=(num+1),dtype=int)
        for i in range(0,num+1):    #creating an array of indices
            numa[i]=i
        np.delete(wdar,0)
        #print(len(numa),len(wdar))
        #print(numa,wdar)
        #plt.scatter(numa,wdar)
        return
        
    def predict(self,X):
        import numpy  
        #print('numpy:', numpy.__version__)
        import numpy as np
        #W=self.W
        w=self.w
        w0=self.w0
        ncls=self.ncls
        cls=self.cls
        
        Ntest=len(X[:,0])       
        ychk=np.zeros(shape=Ntest,dtype=int)+10
        #print(w)
        #print(X)
        #accuracy=0
        for i in range(0,Ntest):
            #print(i)
            #X=Xtest[i]
            cchk=MySVM2.multi(w,w0,X[i])
            if (cchk>=0):
                ychk[i]=cls[0]
            else:
                ychk[i]=cls[1]
            #if (ychk[i]==ytest[i]):
            #accuracy=accuracy+1
            #accuracy=accuracy/Ntest
            #print(accuracy)
        
        return ychk

def my_cross_val(method, X,y,k, mb):
    #import MultiGaussClassify
    from sklearn.linear_model import LogisticRegression
    #import scipy.io as sio
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    #from sklearn.model_selection import cross_val_score
    import numpy  
    #print('numpy:', numpy.__version__)
    import numpy as np
    #print(method)
    if (method==LogisticRegression):
        my = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
    elif (method==MySVM2):
        my=MySVM2(mb)
    #elif (method==MultiGaussClassify):
        #if (diag==False):
            #my=MultiGaussClassify(False)
        #else:
            #my=MultiGaussClassify(True)
    #elif (method==MiltiGaussClassifyDiag):
        #my=MultiGaussClassify(True)
    else: 
        print('not a known method')
    
    Nrow=len(y)
    Ncol=len(X[0,:])
    allmat=np.zeros(shape=(Nrow,Ncol+1), dtype=int)
    for j in range (0,Nrow):
        for l in range (0,Ncol):
            allmat[j,l]=X[j,l]
            allmat[j,Ncol]=y[j]
            
    np.random.shuffle(allmat)
    #print(*allmat[:,8])
    divk=int(Nrow/k)
    remk=int(Nrow%divk)
    #print(Nrow, Ncol, divk, remk)
    Xnew=np.zeros(shape=(Nrow-remk,Ncol), dtype=int)
    ynew=np.zeros(shape=Nrow-remk, dtype=int)
    for j in range (0,Nrow-remk):
        for l in range (0,Ncol):
            Xnew[j,l]=allmat[j,l]
            ynew[j]=allmat[j,Ncol]
    #print(*ynew)
    #for i in range(0,k):
        #print(i)
        #print(*ynew[0+i*divk:(i+1)*divk])
        #print(*Xnew[0+i*divk:(i+1)*divk,5])
    accuracy=np.zeros(k)
    for i in range (0,k):
        #loop=0
        #Xtrain=np.zeros(shape=(Nrow-divk-remk,Ncol))
        #ytrain=np.zeros(shape=Nrow-divk-remk)
        Xtest=np.zeros(shape=(divk,Ncol), dtype=int)
        ytest=np.zeros(shape=divk, dtype=int)
        Xloop=Xnew
        yloop=ynew
        for j in range (0,divk):
            Xtest[j,:]=Xnew[j+i*divk,:]
            ytest[j]=ynew[j+i*divk]
            #print(i,j)
        Xloop=np.delete(Xloop, np.s_[i*divk:(i+1)*divk], axis = 0)
        yloop=np.delete(yloop, np.s_[i*divk:(i+1)*divk], axis = 0)
        #print(i)
        #print(*ytest)
        #print(*Xtest[:,5])
        Xtrain=Xloop
        ytrain=yloop
        #print(i,np.shape(ytrain),np.shape(ytest))
        my.fit(Xtrain, ytrain)
        ypred = my.predict(Xtest)
        count=0
        for j in range (0,divk):
            if ypred[j]==ytest[j]:
                count=count+1
        #print(count,divk)        
        accuracy[i]=count/divk    
        #loop=loop+1
    #accuracy = cross_val_score(my,X,y,cv=k)
    error_rate=np.zeros(k)
    #print(k,np.shape(error_rate), np.shape(accuracy))
    for i in range(0, k):
        error_rate[i]=1.0-accuracy[i]
        #print(i,accuracy[i],error_rate[i])
    print('error rate=',error_rate)
    #print(accuracy)
    print('Mean=', np.mean(error_rate))
    print('Standard Deviation=',np.std(error_rate))
    return 

q3()
