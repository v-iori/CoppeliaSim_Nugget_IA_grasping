## @author: Vincent The Young One
## @modif : Vincent The Old One

import numpy as np
from PIL import Image
from skimage.transform import resize
from os import listdir


#def DemoGen():
support = np.load('demoImgGen/depth_parameters_skeleton.npy')
support_label = support[1]
support_depth = support[0]

nb_traitements = len(listdir("demoImgGen/original"))

for i in range(nb_traitements):
    depth = np.load('demoImgGen/original/aligned_depth_preprocessed_{}.npy'.format(i)).astype('uint8')
    depth_image = Image.fromarray(depth, "RGB")
    depth_rectangle = Image.open("demoImgGen/labels/label{}.png".format(i))

    depth_parameters_demo_label = np.asarray(depth_rectangle)
    depth_parameters_demo_label = depth_parameters_demo_label[0:480, 0:480, 0:3].astype('int64')
    depth_parameters_demo_image = np.asarray(depth_image).astype('int64')
    depth_parameters_demo = np.copy(support)

    for j in range(480):
        for k in range(480):
            if(depth_parameters_demo_label[j, k, 1] > 130 & depth_parameters_demo_label[j, k, 1] < 200):
                depth_parameters_demo_label[j, k, 0] = 1
                depth_parameters_demo_label[j, k, 1] = 158
                depth_parameters_demo_label[j, k, 2] = 56

            elif(depth_parameters_demo_label[j, k, 0] > 20):
                depth_parameters_demo_label[j, k, 0] = -1
                depth_parameters_demo_label[j, k, 1] = 0
                depth_parameters_demo_label[j, k, 2] = 0

            else:
                depth_parameters_demo_label[j, k, 0] = 0
                depth_parameters_demo_label[j, k, 1] = 0
                depth_parameters_demo_label[j, k, 2] = 0

    depth_parameters_demo[0] = depth_parameters_demo_image
    depth_parameters_demo[1] = depth_parameters_demo_label
    np.save('demoImgGen/result/depth_parameters_demo{}'.format(i),depth_parameters_demo)


#DemoGen()