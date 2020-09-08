import os
import random
path = '/home/yashas/Desktop/brain_cnn/cemri_download/all_images'
fold = ['glioma','meningioma','pituitary']


glioma,meningioma,pituitary = [], [], []
x = []

for file in os.listdir(path):

    x.append(file)


random.shuffle(x)
print(len(x))

print(x)
vali = x[:400]
test = x[400:]


for i in vali:
    dest = '/home/yashas/Desktop/brain_cnn/cemri_download/validtion_set/' + i
    source = "/home/yashas/Desktop/brain_cnn/cemri_download/all_images/" +i
    os.rename(source, dest)