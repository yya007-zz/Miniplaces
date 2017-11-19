image = scipy.misc.imread('./1.jpg')
image = scipy.misc.imresize(image, (256,256))
image = image.astype(np.float32)/255.