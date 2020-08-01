
def q3i():
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
    
    
    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import cross_val_score
    # initialize a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    
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
    
    print('1.LinearSVC with Boston50')
    my_cross_val(LinearSVC, Xbos, yb50,10)
    print('\n','2.LinearSVC with Boston75')
    my_cross_val(LinearSVC, Xbos, yb75,10)
    print('\n','3.LinearSVC with Digits')
    my_cross_val(LinearSVC, Xdig, ydig,10)
    print('\n','4.SVC with Boston50')
    my_cross_val(SVC, Xbos, yb50,10)
    print('\n','5.SVC with Boston75')
    my_cross_val(SVC, Xbos, yb75,10)
    print('\n','6.SVC with Digits')
    my_cross_val(SVC, Xdig, ydig,10)
    print('\n','7.LogisticRegression with Boston50')
    my_cross_val(LogisticRegression, Xbos, yb50,10)
    print('\n','8.LogisticRegression with Boston75')
    my_cross_val(LogisticRegression, Xbos, yb75,10)
    print('\n','9.LogisticRegression with Digits')
    my_cross_val(LogisticRegression, Xdig, ydig,10)
    return
    
     


def q3ii(): 
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
    
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
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
    
    print('1.LinearSVC with Boston50')
    my_train_test(LinearSVC, Xbos, yb50,0.75,10)
    print('\n','2.LinearSVC with Boston75')
    my_train_test(LinearSVC, Xbos, yb75,0.75,10)
    print('\n','3.LinearSVC with Digits')
    my_train_test(LinearSVC, Xdig, ydig,0.75,10)
    print('\n','4.SVC with Boston50')
    my_train_test(SVC, Xbos, yb50,0.75,10)
    print('\n','5.SVC with Boston75')
    my_train_test(SVC, Xbos, yb75,0.75,10)
    print('\n','6.SVC with Digits')
    my_train_test(SVC, Xdig, ydig,0.75,10)
    print('\n','7.LogisticRegression with Boston50')
    my_train_test(LogisticRegression, Xbos, yb50,0.75,10)
    print('\n','8.LogisticRegression with Boston75')
    my_train_test(LogisticRegression, Xbos, yb75,0.75,10)
    print('\n','9.LogisticRegression with Digits')
    my_train_test(LogisticRegression, Xdig, ydig,0.75,10)
    return

def q4():
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


    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import cross_val_score
    # initialize a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    
    from sklearn.datasets import load_digits
    digits = load_digits()
    Xdig=digits.data
    ydig=digits.target
    
    X1new=rand_proj(Xdig,32)
    #print(*X1new[5,:])
    
    X2new=quad_proj(Xdig)
    #print('\n',*X2new[5,:])
    
    print('\n','1.LinearSVC with X1')
    my_cross_val(LinearSVC, X1new, ydig,10)
    print('\n','2.LinearSVC with X2')
    my_cross_val(LinearSVC, X2new, ydig,10)
    print('\n','3.SVC with X1')
    my_cross_val(SVC, X1new, ydig,10)
    print('\n','4.SVC with X2')
    my_cross_val(SVC, X2new, ydig,10)
    print('\n','5.Logistic regression with X1')
    my_cross_val(LogisticRegression, X1new, ydig,10)
    print('\n','6.Logistic regression with X2')
    my_cross_val(LogisticRegression, X2new, ydig,10)
    
    return
    

def my_cross_val(method, X,y,k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    #from sklearn.model_selection import cross_val_score
    import numpy  
    #print('numpy:', numpy.__version__)
    import numpy as np
    if (method==LogisticRegression):
        my = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)
    elif (method==SVC):
        #from sklearn.svm import SVC
        my=SVC(gamma='scale',C=10)
    elif (method==LinearSVC):
        my=LinearSVC(max_iter=2000)
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

def my_train_test(method, X,y,pi,k):
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
    #from sklearn.model_selection import train_test_split
    #from sklearn.model_selection import cross_val_score
    # initialize a Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    #from sklearn.metrics import accuracy_score
    accuracy=np.zeros(k)
    error_rate=np.zeros(k)
    Nrow=len(y)
    Ncol=len(X[0,:])
    #print(Nrow,Ncol)
    allmat=np.zeros(shape=(Nrow,Ncol+1), dtype=int)
    for j in range (0,Nrow):
        for l in range (0,Ncol):
            allmat[j,l]=X[j,l]
        allmat[j,Ncol]=y[j]
    np.random.shuffle(allmat)
    #print(*allmat[:,64])
    #Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=None, test_size=pi)
    if (method==LogisticRegression):
        my = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',
                         max_iter=5000)
    elif (method==SVC):
            #from sklearn.svm import SVC
        my=SVC(gamma='scale',C=10)
    elif (method==LinearSVC):
        my=LinearSVC(max_iter=2000)
    else: 
        print('not a known method')
    for i in range (0,k):
        #Nrow=len(y)
        #Ncol=len(X[0,:])
        #allmat=np.zeros(shape=(Nrow,Ncol+1))
        #for j in range (0,Nrow):
         #   for l in range (0,Ncol):
          #      allmat[j,l]=X[j,l]
           #     allmat[j,Ncol]=y[j]
        np.random.shuffle(allmat)
        #print(*allmat[:,8])
        #divk=int(Nrow/k)
        #remk=Nrow%divk
        rowper=int(Nrow*(1-pi))
        #print('rowper=',rowper)
        Xnew=np.zeros(shape=(Nrow,Ncol), dtype=int)
        ynew=np.zeros(shape=Nrow, dtype=int)
        for j in range (0,Nrow):
            for l in range (0,Ncol):
                Xnew[j,l]=allmat[j,l]
                ynew[j]=allmat[j,Ncol]
        #print(*ynew)
        #for i in range(0,k):
        #print(i)
        #print(*ynew[0+i*divk:(i+1)*divk])
        #print(*Xnew[0+i*divk:(i+1)*divk,5])
        accuracy=np.zeros(k)
        #for i in range (0,k):
        #loop=0
        #Xtrain=np.zeros(shape=(Nrow-divk-remk,Ncol))
        #ytrain=np.zeros(shape=Nrow-divk-remk)
        Xtest=np.zeros(shape=(rowper,Ncol), dtype=int)
        ytest=np.zeros(shape=rowper, dtype=int)
        Xloop=Xnew
        yloop=ynew
        #print(i)
        for j in range (0,rowper):
            #print(j)
            Xtest[j,:]=Xnew[j,:]
            ytest[j]=ynew[j]
            #print(i,j)
        Xloop=np.delete(Xloop, np.s_[0:rowper], axis = 0)
        yloop=np.delete(yloop, np.s_[0:rowper], axis = 0)
        #print(i)
        #print('accuracy')
        #print(*ytest)
        #print(*Xtest[:,5])
        Xtrain=Xloop
        ytrain=yloop
        my.fit(Xtrain, ytrain)
        ypred = my.predict(Xtest)
        #for i in range (0,k):
        #print(i,'ytest=',*ytest,'\n','ytrain=',*ytrain)
        #print(np.shape(ytrain),np.shape(ytest))
        count=0
        for j in range (0,rowper):
            if ypred[j]==ytest[j]:
                count=count+1
        #print('count=',count)        
        accuracy[i]=count/rowper
        #print(count,rowper)
        error_rate[i]=1.0-accuracy[i]
        #print('count=',count, 'accuracy', accuracy[i], 'error_rate=',error_rate[i])
        #print(np.shape(accuracy))
        #print(accuracy)
        #my.fit(Xtrain, ytrain)
        #ypred = my.predict(Xtest)
    
    #accuracy=accuracy_score(ytest, ypred)
    #error_rate=np.zeros(k)
    #error_rate[i]=1.0-accuracy
    #accuracy = cross_val_score(my,X,y,cv=k)
    #error_rate=np.zeros(k)
    #print(k,np.shape(error_rate), np.shape(accuracy))
    #for i in range(0, k):
        #error_rate[i]=1.0-accuracy[i]
        #print(i,accuracy[i],error_rate[i])
    print('error rate=',error_rate)
    #print(accuracy)
    print('Mean=', np.mean(error_rate))
    print('Standard Deviation=',np.std(error_rate))
    return 

def quad_proj(X):
    import numpy as np
    Ndig=len(X[:,1])
    Ndigx=len(X[1,:])
    Ntot=int(Ndigx*(Ndigx-1)/2+2*Ndigx)
    X2new=np.zeros(shape=(Ndig,Ntot), dtype=int)
    for i in range (0,Ndig):
        count=0
        #print(i)
        for j in range (0,Ndigx):
            X2new[i,j]=X[i,j]
        for j in range(0,Ndigx):
            X2new[i,j+Ndigx]=X[i,j]*X[i,j]
        for j in range (0,Ndigx):
            for jd in range (j+1, Ndigx):
                X2new[i,2*Ndigx+count]=X[i,j]*X[i,jd]
                count=count+1
    return X2new

def rand_proj(X,d):
    import numpy as np
    mu=0
    sigma=1
    Ndigx=len(X[1,:])
    G = np.random.normal(mu, sigma, size=(Ndigx, d))
    X1new=np.matmul(X,G)
    return X1new


