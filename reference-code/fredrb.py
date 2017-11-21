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
    x=np.zeros([n,126,126,3])
    y=np.zeros([n],dtype=np.int32)
    for i in range(n):
        pos=np.random.randint(0,high=2,size=2)
        name = train[indexes[i]].split()[0]
        category = int(train[indexes[i]].split()[1])
        y[i]=category
        name = "images/" + str(name)
        x[i] = (sm.imread(name)*np.float32(1/255)-mean)[pos[0]:pos[0]+126,pos[1]:pos[1]+126,:]
        flip = np.random.random_integers(0,1)
        if flip>0:
            x[i]=x[i,:,::-1,:]
    return (x,y)

def generate_val_samples(n):
    indexes=np.random.randint(0,high=val_index-1,size=n)
    x=np.zeros([n,126,126,3])
    y=np.zeros([n],dtype=np.int32)
    for i in range(n):
        name = val[indexes[i]].split()[0]
        category = int(val[indexes[i]].split()[1])
        y[i]=category
        name = "images/" + str(name)
        x[i] = (sm.imread(name)*np.float32(1/255)-mean)[1:127,1:127,:]
    return (x,y)

def generate_triple_samples(n):
    indexes=np.random.randint(0,high=index-1,size=n)
    x=np.zeros([n,126,126,3])
    up=np.zeros([n,42,126,3])
    down=np.zeros([n,42,126,3])
    y=np.zeros([n],dtype=np.int32)
    for i in range(n):
        pos=np.random.randint(0,high=2,size=2)
        name = train[indexes[i]].split()[0]
        category = int(train[indexes[i]].split()[1])
        y[i]=category
        name = "images/" + str(name)
        x[i] = (sm.imread(name)*np.float32(1/float(255))-mean)[pos[0]:pos[0]+126,pos[1]:pos[1]+126,:]
        flip = np.random.random_integers(0,1)
        if flip>0:
            x[i]=x[i,:,::-1,:]
        up[i] = x[i][0:42,:,:]
        down[i] = x[i][84:126,:,:]

    return (x,up,down,y)
