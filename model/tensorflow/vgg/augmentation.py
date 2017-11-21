from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def augmentation(img):
	datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

	x = img_to_array(img)
	x = x.reshape((1,) + x.shape)

	i = 0
	for batch in datagen.flow(x, batch_size=1):
		return batch[0]