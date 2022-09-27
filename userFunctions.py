import tensorflow as tf
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.transform import resize
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
#import pyrealsense2 as rs
import cv2
import imageio
import os
import time
import shutil
import math
from scipy import ndimage, misc

def action(trainer, save, snapshot_file, viz, image, i):
    '''
    Perform an action decided by a neural network
    :param trainer: Model
    :param Robot: True will perform grasp in real life, False will perform test on automated_test file
    '''     
    x_pred, y_pred, x_piece_barycentre, y_piece_barycentre, _, out, t1, t2, t3, t4, input_preprocessed, result = get_pred(trainer, save, viz, snapshot_file, i, image)
            
    return  x_pred, y_pred, x_piece_barycentre, y_piece_barycentre, t1, t2, t3, t4, input_preprocessed, result

def get_pred(trainer, save, viz, snapshot_file, i, image):
    if image :
        
        color_output = plt.imread("datasetTest/{}/color/aligned_color_crop{}.png".format(snapshot_file,i)) 
        depth_image_png=Image.open("datasetTest/{}/color/aligned_color_crop{}.png".format(snapshot_file,i)) 
        depth_image = np.asarray(depth_image_png).astype('uint8')
        for j in range(480):
            for k in range(480):
                if(depth_image[j,k,2]>200):
                    depth_image[j,k,0]=255
                    depth_image[j,k,1]=255
                    depth_image[j,k,2]=255
                else:
                    depth_image[j,k,0]=0
                    depth_image[j,k,1]=0
                    depth_image[j,k,2]=0
        image_rgb_bin = Image.fromarray(depth_image, "RGB")
        image_rgb_bin.save('Test_image_coppeliasim_bin_{}.png'.format(i))
        plt.clf()
    else:
        depth_image = np.load("captures/{}/depth_raw/aligned_depth_{}.npy".format(snapshot_file,i))
        color_output = plt.imread("captures/{}/color/aligned_color_crop{}.png".format(snapshot_file,i)) 
        plt.clf()
        # return
    t1 = time.time()    #Acquisition ou chargement de l'image

    init_shape = depth_image.shape
    #depth_image = preprocess_depth_img(depth_image, save, viz, i)
    input_preprocessed = depth_image

    if viz:
        plt.imshow(depth_image)
        plt.show()

    depth_image = resize(depth_image, (224, 224, 3), anti_aliasing=False)
    depth_image = (depth_image * 255).astype('uint8')
    depth_image = depth_image.reshape((1, 224, 224, 3))
    
    t2 = time.time()    #Préprocess de l'image et redimensionnements

    output_prob = trainer.forward(depth_image)
    out_numpy = output_prob[0].numpy()

    t3 = time.time()    #Traitement par le NN et Qmap de sortie

    out = trainer.prediction_viz(out_numpy, depth_image)
    out = out.reshape((224, 224, 3))
    out = resize(out, init_shape)
    result=out
    x_pred, y_pred, x_piece_barycentre, y_piece_barycentre, e_pred, angle_viz = postprocess_pred(out, viz)
    
    t4 = time.time()    #Post-process de la Qmap
    
    
    if viz :
        results = np.zeros(shape=(480,480,3))
        preds = np.zeros(shape=(5))
        results[:,:,:]=result
        x_viz = 320 + (x_pred - 320)/1.33333
        x_viz_barycentre = 320 + (x_piece_barycentre - 320)/1.33333
        x_viz=x_pred
        x_viz_barycentre =  (x_piece_barycentre )
        v = np.array([x_viz_barycentre-x_viz,y_piece_barycentre-y_pred])
        angle_pred = np.arccos(v[0]/norm(v))*np.sign(np.arcsin(v[1]/norm(v)))*180/math.pi
        preds[:]=[x_viz,y_pred,x_viz_barycentre,y_piece_barycentre,angle_pred]
        out2=results[:,:,:]
        out2[:, :, 1] = (out2[:, :, 1] - 0.5) * 2
        desired_output = out2[:, :, 1]*(out2[:, :, 0] != 0).astype(np.int)
        desired_output = resize(desired_output, (480, 480, 1), anti_aliasing=True)
        rect = draw_rectangle(80, preds[4], preds[0]-80, preds[1], 20)
        plt.imshow(desired_output, vmin=-1, vmax=1, cmap='RdYlGn')
        plt.colorbar(label='Value of Output')
        plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='blue')
        plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='blue')
        plt.scatter(preds[0]-80, preds[1], color='blue')
        plt.scatter(preds[2]-80, preds[3], color='purple')
        plt.show()   
        plt.clf()
        plt.imshow(color_output, vmin=-1, vmax=1, cmap='RdYlGn')
        plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='blue')
        plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='blue')
        plt.scatter(preds[0]-80, preds[1], color='blue')
        plt.scatter(preds[2]-80, preds[3], color='purple')
        plt.show()   
        plt.clf()

    return x_pred, y_pred, x_piece_barycentre, y_piece_barycentre, e_pred, out, t1, t2, t3, t4, input_preprocessed, result

def make_clean_folder(path_folder):
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        os.makedirs(path_folder)

def learning(demo_depth_label, trainer,epoch):
    trainer.main_online(demo_depth_label, epoch)
    return trainer

def load(snapshot_file):
    demo_depth_label = []
    demo_depth_label_file = [join("datasetDemo/{}/".format(snapshot_file), f) for f in
                             listdir("datasetDemo/{}/".format(snapshot_file)) if
                             isfile(join("datasetDemo/{}/".format(snapshot_file), f))]
    for f in demo_depth_label_file:
        print('Loading : ', f)
        demo_depth_label.append(np.load(f))
    return demo_depth_label

def heatmap2pointcloud(img):
    PointCloudList = []
    for index, x in np.ndenumerate(img):
        for i in range(int(x*100)):
            PointCloudList.append([index[1], img.shape[0]-index[0]])
    PointCloudList = np.array(PointCloudList)
    return np.asarray(PointCloudList)

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))

    det = v1[0] * v2[1] - v1[1] * v2[0]

    if det < 0:
        return -np.arctan2(sinang, cosang)
    else:
        return np.arctan2(sinang, cosang)

def preprocess_depth_img(depth_image, save, viz, i):
    seuil = 8
    seuillage = np.amax(depth_image) - seuil
    depth_image[depth_image > seuillage ] = 0
    depth_image[depth_image != 0] = 1
    depth_image = depth_image[0:480,80:560]
    depth_image = (depth_image * 255).astype('uint8')
    depth_image = np.asarray(np.dstack((depth_image, depth_image, depth_image)), dtype=np.uint8)
    return depth_image

def rotate_image2(input_data, input_angles):
    return tf.contrib.image.rotate(input_data, input_angles, interpolation="BILINEAR")

def preprocess_img(img, target_height=224*5, target_width=224*5, rotate=False):
    # Apply 2x scale to input heightmaps
    resized_img = tf.image.resize(img, (target_height, target_width))
    # Peut être rajouter un padding pour éviter les effets de bords
    return resized_img

def postprocess_img( imgs, list_angles):
    # Return Q values (and remove extra padding
    # Reshape to standard
    resized_imgs = tf.image.resize(imgs, (320, 320))
    # Perform rotation
    rimgs = rotate_image2(resized_imgs, list_angles)
    # Reshape rotated images
    resized_imgs = tf.image.resize(rimgs, (320, 320))
    return resized_imgs

def postprocess_pred(out, viz):
    out[:, :, 1] = out[:, :, 1]*(out[:, :, 0] != 0).astype(np.int)
    (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
    
    fenetre_barycentre = 200
    pourcent_seuillage = 0.5
    Borders=bords(out[:,:,1],x_max,y_max,fenetre_barycentre)
    fenetre=reduction_fenetre(out[:,:,1],x_max,y_max,Borders)
    seuillage=0.5+(out[y_max,x_max,1]-0.5)*pourcent_seuillage

    x_pred_barycentre=barycentre(fenetre,seuillage)[0]+Borders[2]
    y_pred_barycentre=barycentre(fenetre,seuillage)[1]+Borders[0]

    x_piece_barycentre=barycentre(out[:,:,1],0.01)[0]
    y_piece_barycentre=barycentre(out[:,:,1],0.01)[1]

    v = np.array([x_piece_barycentre-x_pred_barycentre,y_piece_barycentre-y_pred_barycentre])
    angle_viz = np.arccos(v[0]/norm(v))*np.sign(np.arcsin(v[1]/norm(v)))*180/math.pi
    e = 80

    return x_pred_barycentre, y_pred_barycentre, x_piece_barycentre, y_piece_barycentre, e, angle_viz

def draw_rectangle(e, theta, x0, y0, lp):
    theta_rad = - theta * np.pi/180
    x1 = int(x0 - lp/2*np.cos(theta_rad) - e/2*np.sin(theta_rad))
    y1 = int(y0 + lp/2*np.sin(theta_rad) - e/2*np.cos(theta_rad))
    x2 = int(x0 + lp/2*np.cos(theta_rad) - e/2*np.sin(theta_rad))
    y2 = int(y0 - lp/2*np.sin(theta_rad) - e/2*np.cos(theta_rad))
    x3 = int(x0 - lp/2*np.cos(theta_rad) + e/2*np.sin(theta_rad))
    y3 = int(y0 + lp/2*np.sin(theta_rad) + e/2*np.cos(theta_rad))
    x4 = int(x0 + lp/2*np.cos(theta_rad) + e/2*np.sin(theta_rad))
    y4 = int(y0 - lp/2*np.sin(theta_rad) + e/2*np.cos(theta_rad))

    return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int)

def evenisation(x):
    if x%2 == 0:
        return x
    else:
        return x+1

def compute_labels(best_pix_ind, shape=(224,224,3), viz=False):
    '''Create the targeted Q-map
    :param label_value: Reward of the action
    :param best_pix_ind: (Rectangle Parameters : x(colonne), y(ligne), angle(en degré), ecartement(en pixel)) Pixel where to perform the action
    :return: label : an 224x224 array where best pix is at future reward value
             label_weights : a 224x224 where best pix is at one

    label is the a 224x224x3 array where
        - First Channel is label
        - Second Channel is the angle
        - Third Channel is the spreading
    '''
    label = np.zeros(shape, dtype=np.float32)
    print(best_pix_ind)
    for i in range(len(best_pix_ind)):
        label_temp = np.zeros(shape[:2], dtype=np.float32)
        angle_temp = np.zeros(shape[:2], dtype=np.float32)
        ecart_temp = np.zeros(shape[:2], dtype=np.float32)

        x, y, angle, e, lp, label_val = best_pix_ind[i]
        rect = draw_rectangle(e, angle, x, y, lp)
        cv2.fillConvexPoly(label_temp, rect, color=(255))
        print(label_temp.shape)
        if label_val==1:
            print('ici')
            angle_temp[np.where(label_temp == 255)] = angle
            print(angle)
            ecart_temp[np.where(label_temp == 255)] = e
        label_temp[np.where(label_temp == 255)] = label_val
        label[:, :, 0] = label[:, :, 0] + label_temp
        label[:, :, 1] = label[:, :, 1] + angle_temp
        label[:, :, 2] = label[:, :, 2] + ecart_temp

    label = label.astype(np.int)
    return label

best_pix_ind = [[100, 200, 45, 20, 20, 255], [200, 100, 45, 20, 20, 150]]
compute_labels(best_pix_ind)


# def record_depth(save,i,profile, pipeline):
#     # if i==0:
#     #         make_clean_folder("realshot/")
#     # Getting the depth sensor's depth scale (see rs-align example for explanation)

#     depth_sensor = profile.get_device().first_depth_sensor()
#     depth_scale = depth_sensor.get_depth_scale()

#     # We will be removing the background of objects more than
#     #  clipping_distance_in_meters meters away
#     clipping_distance_in_meters = 1  # 1 meter
#     clipping_distance = clipping_distance_in_meters / depth_scale

#     # Create an align object
#     # rs.align allows us to perform alignment of depth frames to others frames
#     # The "align_to" is the stream type to which we plan to align depth frames.
#     align_to = rs.stream.color
#     align = rs.align(align_to)

#     # Get frameset of color and depth
#     frames = pipeline.wait_for_frames()

#     # frames.get_depth_frame() is a 640x360 depth image

#     # Align the depth frame to color frame
#     aligned_frames = align.process(frames)

#     # Get aligned frames
#     # aligned_depth_frame is a 640x480 depth image
#     aligned_depth_frame = aligned_frames.get_depth_frame()
#     color_frame = aligned_frames.get_color_frame()
#     # np.save("realshot/aligned_depth_{}.npy".format(i), aligned_depth_frame.get_data())
#     if save :
#         np.save("realshot/aligned_depth_{}.npy".format(i), aligned_depth_frame.get_data())
#         np.save("realshot/aligned_color_{}.npy".format(i), color_frame.get_data())
#         color_frame_crop = np.asarray(color_frame.get_data())
#         color_frame_crop = color_frame_crop[0:480,80:560]
#         color_frame_crop = color_frame_crop[:,:,::-1]
#         image_calib=Image.fromarray(color_frame_crop,"RGB")
#         image_calib.save("realshot/color_{}.png".format(i))
        
#     # pipeline.stop()

#     image_npy = np.asarray(aligned_depth_frame.get_data())
#     color = np.asarray(color_frame.get_data())
#     return image_npy, color

def barycentre (array,seuillage):

    copie = np.copy(array)
    copie[copie>seuillage]=1
    copie[copie<seuillage]=0
    nonZeroMasses=np.nonzero(copie)
    barycentre_x = np.average(nonZeroMasses[1])
    barycentre_y = np.average(nonZeroMasses[0])
    
    return [np.around(barycentre_x).astype(int),np.around(barycentre_y).astype(int)]

def bords(array,x_pred,y_pred,pixel):
    
    Haut=max(y_pred-pixel,0)
    Bas=min(y_pred+pixel,len(array[:,0]))
        
    Gauche=max(x_pred-pixel,0)
    Droite=min(x_pred+pixel,len(array[0,:]))
    
    return [Haut,Bas,Gauche,Droite]

def reduction_fenetre(array,x_pred,y_pred,borders):
    
    new_array=array[borders[0]:borders[1],borders[2]:borders[3]]
    
    return new_array