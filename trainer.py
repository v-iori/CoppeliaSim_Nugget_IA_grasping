# -*- coding: utf-8 -*-

import model as mod
import tensorflow as tf
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import userFunctions as div
import cv2
import dataAugmentation as da
#import tfmpl                             # Put matplotlib figures in tensorboard
import os
from scipy import ndimage, misc

class Trainer(object):
    def __init__(self, savetosnapshot=True, load=False, snapshot_file='name',cnn=0):
        super(Trainer, self).__init__()
        self.cnn=cnn
        self.myModel = mod.FullNN(self.cnn)
        self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
        self.width, self.height = 0, 0
        self.best_idx, self.future_reward = [0, 0], 0
        self.scale_factor = 1
        self.output_prob = 0
        self.loss_value = 0
        self.iteration = 0
        self.num = 0
        self.classifier_boolean = False
        self.savetosnapshot = savetosnapshot

        # Frequency
        self.viz_frequency = 200
        self.saving_frequency = 199
        self.dataAugmentation = da.OnlineAugmentation()
        # Initiate logger
        checkpoint_directory = "savedCheckpoint/{}/".format(snapshot_file)
        if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.myModel,
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())
        if savetosnapshot:
            self.snapshot_file = os.path.join(checkpoint_directory, snapshot_file)
            self.checkpoint.save(self.snapshot_file)
            print('Creating snapshot : {}'.format(self.snapshot_file))
        if load:
            latest_snapshot_file = tf.compat.v1.train.latest_checkpoint(checkpoint_directory)
            self.checkpoint.restore(latest_snapshot_file)
            print('Pre-trained model snapshot loaded from: {}'.format(latest_snapshot_file))

        self.loss = tf.compat.v1.losses.huber_loss

        # Initialize the network with a fake shot
        self.forward(np.zeros((1, 224, 224, 3), np.float32))
        self.vars = self.myModel.trainable_variables

    def forward(self, input):
        self.image = input
        # Increase the size of the image to have a relevant output map
        input = div.preprocess_img(input, target_height=self.scale_factor*224, target_width=self.scale_factor*224)
        # Pass input data through model
        self.output_prob = self.myModel(input)
        self.batch, self.width, self.height = self.output_prob.shape[0], self.output_prob.shape[1], self.output_prob.shape[2]
        # Return Q-map
        return self.output_prob

    def compute_loss_dem(self, label, noBackprop=False):
        for l, output in zip(label, self.output_prob):
            l = self.reduced_label(l)[0]

            l_numpy = l.numpy()
            l_numpy_pos = l_numpy[:, :, 0]
            l_numpy_pos[l_numpy_pos > 10] = -1
            l_numpy[:, :, 0] = l_numpy_pos
            l = tf.convert_to_tensor(l_numpy)

            ### Test Sans poids
            weight = np.abs(output.numpy())
            weight[l_numpy > 0] += 20/(np.sqrt(np.sum(l_numpy > 0)+1))       #Initialement 2.
            weight[l_numpy < 0] += 20/(np.sqrt(np.sum(l_numpy < 0)+1))       #Initialement 1.
            weight[l_numpy == 0] += 1/(np.sqrt(np.sum(l_numpy == 0)+1))      #Initialement 0.2

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars])

            self.loss_value = self.loss(l, output, weight) + 0.00005 * lossL2

            new_lab = l.numpy()
            new_lab = new_lab.reshape((1, *new_lab.shape))

        if (self.savetosnapshot)\
                and (tf.compat.v1.train.get_global_step() is not None)\
                and (tf.compat.v1.train.get_global_step().numpy()%self.saving_frequency == 0)\
                and (not noBackprop):

            self.save_model()
        return self.loss_value

    def compute_labels(self, label_value, best_pix_ind, shape=(224,224,3), viz=True):
        '''Create the targeted Q-map
        :param label_value: Reward of the action
        :param best_pix_ind: (Rectangle Parameters : x(colonne), y(ligne), angle(en degré), ecartement(en pixel)) Pixel where to perform the action
        :return: label : an 224x224 array where best pix is at future reward value
                 label_weights : a 224x224 where best pix is at one
        '''
        # Compute labels
        x, y, angle, e, lp = best_pix_ind
        rect = div.draw_rectangle(e, angle, x, y, lp)
        label = np.zeros(shape, dtype=np.float32)

        cv2.fillConvexPoly(label, rect, color=1)
        label *= label_value

        return label

    def reduced_label(self, label):
        '''Reduce label Q-map to the output dimension of the network
        :param label: 224x224 label map
        :return: label and label_weights in output format
        '''

        label = tf.convert_to_tensor(label, np.float32)
        label = tf.image.resize(label, (self.width, self.height))
        label = tf.reshape(label[:, :, 0], (self.batch, self.width, self.height, 1))

        if self.classifier_boolean:
            label = label.numpy()
            label[label > 0.] = 1
            label = tf.convert_to_tensor(label, np.float32)
        return label

    def main_batches(self, im, label):
        self.future_reward = 1
        # On change le label pour que chaque pixel sans objet soit considéré comme le background
        if type(im).__module__ != np.__name__:
            im_numpy, label_numpy = im.numpy(), label.numpy()
        else:
            im_numpy, label_numpy = im, label
        mask = (im_numpy>20).astype(np.int32)
        label_numpy = label_numpy * mask
        label = tf.convert_to_tensor(label_numpy, dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.myModel.trainable_variables)
            self.forward(im)
            self.compute_loss_dem(label)
            grad = tape.gradient(self.loss_value, self.myModel.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.myModel.trainable_variables),
                                           global_step=tf.compat.v1.train.get_or_create_global_step())
            self.iteration = tf.compat.v1.train.get_global_step()

    def main_online(self, demo_depth_label, nb_batch):
        self.dataAugmentation.create_database(demo_depth_label)
        for batch in tqdm.tqdm(range(nb_batch)):
            batch_im, batch_lab = self.dataAugmentation.get_pair()
            self.main_batches(tf.stack([batch_im]), tf.stack([batch_lab]))

#### Accessoires #######
    def vizualisation(self, img, idx):
        prediction = cv2.circle(img[0], (int(idx[1]), int(idx[0])), 7, (255, 255, 255), 2)
        plt.imshow(prediction)
        plt.show()

    def save_model(self):
        print("Saving model to {}".format(self.snapshot_file))
        self.checkpoint.save(self.snapshot_file)

    def prediction_viz(self, qmap, im):
        qmap = (qmap + 1) / 2
        
        qmap = tf.image.resize(qmap, (4*self.width, 4*self.height))
        qmap = tf.image.resize(qmap, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        qmap = tf.reshape(qmap[:, :, 0], (224, 224))

        x_map, y_map = np.argmax(np.max(qmap, axis=1)), np.argmax(np.max(qmap, axis=0))

        rescale_qmap = qmap
        img = np.zeros((224, 224, 3))
        img[:, :, 0] = im[0, :, :, 0] / np.max(im[0, :, :, 0])
        img[:, :, 1] = rescale_qmap
        img[x_map-5:x_map+5, y_map-5:y_map+5, 2] = 1
        img = img   # To resize between 0 and 1 : to display with output probability
        return img

