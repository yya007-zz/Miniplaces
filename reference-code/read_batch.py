import numpy as np
import scipy.misc as sm

f = open('train.txt', 'r')
train={}
index=0
for line in f:
    train[index]=line
    index+=1
f.close()

ff= open('val.txt', 'r')
val={}
val_index=0
for line in ff:
    val[val_index]=line
    val_index+=1
ff.close()

mean=np.asarray([0.45834960097,0.4464252445,0.41352266842])

def generate_samples(n):
    indexes=np.random.randint(0,high=index-1,size=n)
    x=np.zeros([n,128,128,3])
    y=np.zeros([n],dtype=np.int32)
    for i in range(n):
        name = train[indexes[i]].split()[0]
        category = int(train[indexes[i]].split()[1])
        y[i]=category
        name = "images/" + str(name)
        x[i] = sm.imread(name)*np.float32(1.0/float(255))-mean
        flip = np.random.random_integers(0,1)
        if flip>0:
            x[i]=x[i,:,::-1,:]
    return (x,y)

def generate_val_samples(n):
    indexes=np.random.randint(0,high=val_index-1,size=n)
    x=np.zeros([n,128,128,3])
    y=np.zeros([n],dtype=np.int32)
    for i in range(n):
        name = val[indexes[i]].split()[0]
        category = int(val[indexes[i]].split()[1])
        y[i]=category
        name = "images/" + str(name)
        x[i] = sm.imread(name)*np.float32(1.0/float(255))-mean
    return (x,y)
