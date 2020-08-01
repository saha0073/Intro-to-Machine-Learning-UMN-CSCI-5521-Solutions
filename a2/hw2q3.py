
def hw2q3(): 
    #from __future__ import print_function
    import IPython
    #print('IPython:', IPython.__version__)
    import numpy  
    #print('numpy:', numpy.__version__)
    import numpy as np
    import scipy
    #print('scipy:', scipy.__version__)
    import matplotlib
    #print('matplotlib:', matplotlib.__version__)
    import sklearn
    #print('scikit-learn:', sklearn.__version__)
    

    # initialize a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    #from sklearn.metrics import accuracy_score
    
    from sklearn.datasets import load_boston
    boston = load_boston()
    Xbos=boston.data
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
    ydig=digits.target
    
    print('1. MultiGaussClassify with full covariance matrix on Boston50')
    my_cross_val(MultiGaussClassify, Xbos, yb50,5, False)
    print('\n','2.MultiGaussClassify with full covariance matrix on Boston75')
    my_cross_val(MultiGaussClassify, Xbos, yb75,5,False)
    print('\n','3.MultiGaussClassify with full covariance matrix on Digits')
    my_cross_val(MultiGaussClassify, Xdig, ydig,5,False)
    print('\n','4.MultiGaussClassify with diagonal covariance matrix on Boston50')
    my_cross_val(MultiGaussClassify, Xbos, yb50,5,True)
    print('\n','5.MultiGaussClassify with diagonal covariance matrix on Boston75')
    my_cross_val(MultiGaussClassify, Xbos, yb75,5,True)
    print('\n','6.MultiGaussClassify with diagonal covariance matrix on Digits')
    my_cross_val(MultiGaussClassify, Xdig, ydig,5,True)
    print('\n','7.LogisticRegression with Boston50')
    my_cross_val(LogisticRegression, Xbos, yb50,5, False)
    print('\n','8.LogisticRegression with Boston75')
    my_cross_val(LogisticRegression, Xbos, yb75,5, False)
    print('\n','9.LogisticRegression with Digits')
    my_cross_val(LogisticRegression, Xdig, ydig,5, False)
    return




def my_cross_val(method, X,y,k, diag):
    #import MultiGaussClassify
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    import numpy  
    #print('numpy:', numpy.__version__)
    import numpy as np
    if (method==LogisticRegression):
        my = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
    elif (method==MultiGaussClassify):
        if (diag==False):
            my=MultiGaussClassify(False)
        else:
            my=MultiGaussClassify(True)
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



class MultiGaussClassify:
    #pass
    
    def __init__(self,diag):
        self.diag=diag
        import numpy  
        #print('numpy:', numpy.__version__)
        import numpy as np
        #print(d)
        #a=5
        #self.diagbool=diagbool
        #pass
        #self.first=first
        #self.last=last
        #self.pay=pay
        #self.email=first + '.' + last + '@company.com'
        #self.Pc=np.zeros(k)+(1/k)
        #self.m=np.zeros(shape=(k,d))
        #self.s=np.zeros(shape=(k,d,d))
        #for i in range (0,k):
            #self.s[i]=np.identity(d,d)
        #corrmat=np.identity(Ncol)
        
    def fit(self, X,y):
        import numpy  
        #print('numpy:', numpy.__version__)
        import numpy as np
        #np.set_printoptions(precision=4)
        Xtrain=X
        ytrain=y
        #pi=.25
        
        #Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata,ydata,random_state=None, test_size=pi)
        Ncol=len(Xtrain[1,:])
        #Ncol=1
        Nrow=len(ytrain)
        
        
        #min1=min(ytrain)
        #ncls=1
        cls=np.array([ytrain[0]])
        
        for i in range (1,Nrow):
            #print(i)
            ncls=np.size(cls)
            count=0
            for j in range (0,ncls):
                if (ytrain[i]!=cls[j]):
                    count=count+1
            if (count==ncls):
                cls=np.append(cls,ytrain[i])
        
        #print(cls)    
        ncls=np.size(cls)   #total number of class
        #print(ncls)
        
        r=np.zeros(shape=(ncls,Nrow), dtype=int)+50
        #r=[10]*Ntest 
        Ptot=0
        Pc=np.zeros(ncls)
        for x in range (0,ncls):
            for i in range (0,Nrow):
                if (ytrain[i]==cls[x]):
                    r[x,i]=1.0
                    Pc[x]=Pc[x]+1
                else:
                    r[x,i]=0.0
            Pc[x]=Pc[x]/Nrow
            Ptot=Pc[x]+Ptot
        #print(Pc)
        #print(Ptot)
        #print(self.d)    
        mn=np.zeros(shape=(ncls,Ncol))
        md=np.zeros(shape=(ncls))
        m=np.zeros(shape=(ncls,Ncol))
        sn=np.zeros(shape=(ncls,Ncol,Ncol))
        sd=np.zeros(shape=(ncls))
        s=np.zeros(shape=(ncls,Ncol,Ncol))
        
        for x in range (0,ncls):
            for i in range (0,Nrow):
                mn[x]=mn[x]+Xtrain[i]*r[x,i]
                md[x]=md[x]+r[x,i]
                #m0n=m0n+Xtrain[i,:]*(1-ytrain[i])
                #m0d=m0d+(1-ytrain[i])
            m[x]=mn[x]*(1/md[x])
            #m0=m0n*(1/m0d)
            #s1n=np.zeros(shape=(Ncol,Ncol))
            #allmat=np.zeros(shape=(Nrow,Ncol+1), dtype=int)
            #s1d=0
            #s0n=np.zeros(shape=(Ncol,Ncol))
            #s0d=0
            for i in range (0,Nrow):
                sn[x]=sn[x]+np.matmul(np.array([Xtrain[i]-m[x]]).T,np.array([Xtrain[i]-m[x]]))*r[x,i]
                sd[x]=sd[x]+r[x,i]
            s[x]=sn[x]*(1/sd[x])
        
        #correcting s if that is a singular matrix   
        
        corr=0.0001
        if (np.linalg.det(s[x])==0):
            for x in range (0,ncls):
                corrmat=np.identity(Ncol)
                s[x]=s[x]+corr*corrmat
              
        
          
        #considering only diagonal elements of s
        if (self.diag==True):
            #print('True')
            for x in range (0,ncls):
                for i in range (0,Ncol):
                    for j in range (0,Ncol):
                        if (i!=j):
                            s[x,i,j]=0
        
        W=np.zeros(shape=(ncls,Ncol,Ncol))
        w=np.zeros(shape=(ncls,Ncol))
        wx0=np.zeros(ncls)
        for x in range (0,ncls):
            #print(*s[x])
            W[x]=-(1/2)*np.linalg.inv(s[x])
            #W0=-(1/2)*np.linalg.inv(s0)
            w[x]=np.matmul(np.linalg.inv(s[x]),m[x])
            #w0=np.matmul(np.linalg.inv(s0),m0)
            wx0[x]=-(1/2)*np.matmul(np.array([m[x]]),np.array([w[x]]).T)-(1/2)*np.log(np.linalg.det(s[x]))+np.log(Pc[x])
        #k=ncls
        #d=Ncol
        
        #print(*ychk)
        self.W=W
        self.w=w
        self.wx0=wx0
        self.ncls=ncls
        self.cls=cls
                             
                
    def predict (self,X):
        import numpy  
        #print('numpy:', numpy.__version__)
        import numpy as np
        W=self.W
        w=self.w
        wx0=self.wx0
        ncls=self.ncls
        cls=self.cls
        
        Ntest=len(X[:,0])
        #Xchk=X
        #Ntest=
        #ncls=k
        g=np.zeros(shape=(ncls,Ntest))
        #g0=np.zeros(Ntest)
        gtot=np.zeros(shape=(Ntest))
        Px=np.zeros(shape=(ncls,Ntest))
        #P_xC0=np.zeros(Ntest)
        accuracy=0
        #ychk=np.random.normal(0, 1, size=(Ntest))
        ychk=np.zeros(Ntest)+10
        
          #
        for i in range(0,Ntest):
            Xtest=X[i]
            #X=Xtest[i]
            Pmax=0
            for z in range (0,ncls):
                #print('W-', np.shape(W),'X-',np.shape(X),'w-',np.shape(w),'wx0-',np.shape(wx0))
                g[z,i]=np.matmul(Xtest,np.array([np.matmul(W[z],Xtest)]).T)+np.matmul(w[z],np.array([Xtest]).T)+wx0[z]
                #g[z,i]=(X*((W[z]*X)))+(w[z]*(X))+wx0[z]
                gtot[i]=gtot[i]+g[z,i]
                #gtot[i]=1
                #print(i,gtot[i])
            for z in range (0,ncls):
                #Px[z,i]=g[z,i]*(1/gtot[i])
                Px[z,i]=g[z,i]
                #print(X,cls[z],Px[z,i])
            Pmax=max(Px[:,i])
            for z in range (0,ncls):
                if (Px[z,i]==Pmax):
                    ychk[i]=cls[z]
        return ychk
                  
hw2q3()  

