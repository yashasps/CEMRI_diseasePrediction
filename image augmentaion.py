import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


for file in os.listdir('/home/yashas/Desktop/brain_cnn/sep/mod_dim_png/'):
    dispath = '/home/yashas/Desktop/brain_cnn/sep/mod_dim_png/'
    soupath = '/home/yashas/Desktop/brain_cnn/sep/mod_dim_png/'



    data = img_to_array(load_img(soupath+file))
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(rotation_range=24,brightness_range=[0.8,1],zoom_range=[0.5,1.0])
    it = datagen.flow(samples, batch_size=1)

    image = it[0].astype('uint8')
    image = image.squeeze()
    pyplot.imshow(image)
    #pyplot.show()
    pyplot.savefig(dispath + file.split('.')[0] +'_3_aug.png')
