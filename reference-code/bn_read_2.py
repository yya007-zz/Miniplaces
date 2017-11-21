import numpy as np
import scipy.misc as sm
from PIL import Image, ImageEnhance

np.random.seed()

mean=np.asarray([0.45834960097,0.4464252445,0.41352266842])

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

        im =Image.open(name)
        im = ImageEnhance.Color(im).enhance(0.5 + 1.5 * np.random.random_sample())
        im = ImageEnhance.Brightness(im).enhance(0.5 + 1.5 * np.random.random_sample())
        im = ImageEnhance.Contrast(im).enhance(0.5 + 1.5 * np.random.random_sample())
        im = ImageEnhance.Sharpness(im).enhance(3.0 * np.random.random_sample())
        im = np.array(im, dtype=np.uint8)
        x[i] = (im*np.float32(1.0/255.0)-mean)[pos[0]:pos[0]+126,pos[1]:pos[1]+126,:]
        
        #x[i] = (sm.imread(name)*np.float32(1.0/255.0)-mean)[pos[0]:pos[0]+126,pos[1]:pos[1]+126,:]
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
        x[i] = (sm.imread(name)*np.float32(1.0/255.0)-mean)[1:127,1:127,:]
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
        x[i] = (sm.imread(name)*np.float32(1.0/255.0)-mean)[pos[0]:pos[0]+126,pos[1]:pos[1]+126,:]
        flip = np.random.random_integers(0,1)
        if flip>0:
            x[i]=x[i,:,::-1,:]
        up[i] = x[i][0:42,:,:]
        down[i] = x[i][84:126,:,:]

    return (x,up,down,y)

def generate_test_sample(n):
    n+=1
    seq=str(n)
    while len(seq)<5:
        seq="0"+seq
    name="images/test/000"+seq+".jpg"
    batch=np.zeros([18,126,126,3])
    index=0
    for x in range(3):
        for y in range(3):
            for flip in range(2):
                batch[index] = (sm.imread(name)*np.float32(1.0/255.0)-mean)[x:x+126,y:y+126,:]
                if flip ==1 :
                    batch[index]=batch[index,:,::-1,:]
                index+=1
    return (name[7:],batch)
