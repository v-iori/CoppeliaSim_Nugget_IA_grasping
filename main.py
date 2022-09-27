# Import des classes utiles
import sys #os
import time
import tkinter as tk
import traceback
import numpy as np
from numpy.linalg import norm
from math import * 
import matplotlib.pyplot as plt
from skimage.transform import resize
from os import listdir

# Import des différentes classes
import userFunctions as div
from trainer import Trainer
import demoGenerator as DM
# Import de l'argparse
import argparse

#Import controle robot
import os
import threading, time
import signal
from PIL import Image


def launch(args):
    global g_node
    signal.signal(signal.SIGINT, signal_handler)
    
    snapshot_file = args.snapshot_file
    save = args.save
    viz = args.viz
    image = args.image
    training = args.training
    grasp = args.grasp
    cnn = args.cnn
    epoch = args.epoch
    
    try:
        if training :
            demo_depth_label = div.load(snapshot_file=snapshot_file)
            trainer = Trainer(savetosnapshot=True, load=False, snapshot_file=snapshot_file,cnn=cnn)
            trainer = div.learning(demo_depth_label, trainer, epoch=epoch)
        else :
            trainer = Trainer(savetosnapshot=False, load=True, snapshot_file=snapshot_file,cnn=cnn)
        
    except Exception as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        pass

    except RuntimeError as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
    
    if viz==False and training==False:
        div.make_clean_folder("results/{}/".format(snapshot_file))
        div.make_clean_folder("results/{}/depth/".format(snapshot_file))
        div.make_clean_folder("results/{}/color/".format(snapshot_file))
        div.make_clean_folder("results/{}/result/".format(snapshot_file))
    
    K=np.asarray([[600.8204,0,0],[0.1826,603.8160,0],[321.4537,229.5616,1]])
    R=np.asarray([[1,0.0057,-0.0068],[-0.0058,1,-0.0050],[0.0068,0.0050,1]])
    Tr=np.asarray([[-71.2298], [-56.3285], [451.3756]])

    New_Pose=np.zeros((3,3))
    New_Pose[0,0]=R[0,0]
    New_Pose[0,1]=R[0,1]
    New_Pose[0,2]=R[0,2]
    New_Pose[1,0]=R[1,0]
    New_Pose[1,1]=R[1,1]
    New_Pose[1,2]=R[1,2]
    New_Pose[2,0]=Tr[0]
    New_Pose[2,1]=Tr[1]
    New_Pose[2,2]=Tr[2]
    
    t_form=np.dot(New_Pose,K)

    inv_t_form = np.linalg.inv(t_form)
    
    if (training==False):
        if image==False:
            nb_traitements = len(listdir("captures/{}/depth_raw/".format(snapshot_file)))
                  
        else:
            nb_traitements = len(listdir("datasetTest/{}/color/".format(snapshot_file)))
            print(nb_traitements)     
           
        timers = np.zeros(shape=(nb_traitements,8))
        print(nb_traitements) 
        inputs = np.zeros(shape=(480,480,3,nb_traitements))
        colors = np.zeros(shape=(480,480,3,nb_traitements))
        results = np.zeros(shape=(480,480,3,nb_traitements))
        preds = np.zeros(shape=(nb_traitements,5))
        for i in range(nb_traitements):
            print("Traitement de l'image n° : ",i+1)
            x_pred, y_pred, x_piece_barycentre, y_piece_barycentre, t1, t2, t3, t4, input_preprocessed, result = div.action(trainer, save, snapshot_file, viz, image, i)

            #x_pred = 320 + (x_pred - 320)/1.33333
            x_pred =x_pred 
            #x_piece_barycentre = 320 + (x_piece_barycentre - 320)/1.33333
            x_piece_barycentre = x_piece_barycentre

            v = np.array([x_piece_barycentre-x_pred,y_piece_barycentre-y_pred])
            angle_pred = np.arccos(v[0]/norm(v))*np.sign(np.arcsin(v[1]/norm(v)))*180/pi
            print([angle_pred+90,(x_pred-240)*0.15/480,(y_pred-240)*0.15/480])
            Pixel_coordinates=np.asarray([[x_pred,y_pred,1]])

            World_coordinates=pointsToWorld(inv_t_form,Pixel_coordinates)

            x_world=World_coordinates[0,0]
            y_world=World_coordinates[0,1]

            x_target = x_world
            y_target = y_world

            c_target = 90 + angle_pred
            x_target += 12*sin(radians(c_target))
            y_target += - 12*cos(radians(c_target))
                
            inputs[:,:,:,i]=input_preprocessed
            # colors[:,:,:,i]=color
            results[:,:,:,i]=result
            preds[i,:]=[x_pred,y_pred,x_piece_barycentre,y_piece_barycentre,angle_pred]

            # pipeline.stop()
        if viz==False:
            for i in range(nb_traitements):
                save_depth = Image.fromarray(inputs[:,:,:,i].astype('uint8'),"RGB")
                save_depth.save("results/{}/depth/{}_input_preprocessed.png".format(snapshot_file,i))
                out=results[:,:,:,i]
                out[:, :, 1] = (out[:, :, 1] - 0.5) * 2
                desired_output = out[:, :, 1]*(out[:, :, 0] != 0).astype(np.int)
                desired_output = resize(desired_output, (480, 480, 1), anti_aliasing=True)
                rect = div.draw_rectangle(80, preds[i,4], preds[i,0], preds[i,1], 20)
                plt.imshow(desired_output, vmin=-1, vmax=1, cmap='RdYlGn')
                plt.colorbar(label='Value of Output')
                plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='blue')
                plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='blue')
                plt.scatter(preds[i,0], preds[i,1], color='blue')
                plt.scatter(preds[i,2], preds[i,3], color='purple')
                plt.savefig('results/{}/result/{}_result.png'.format(snapshot_file,i), dpi=120)  
                plt.clf()
                if image==False:
                    color_output = plt.imread("captures/{}/color/aligned_color_crop{}.png".format(snapshot_file,i)) 
                else:
                    color_output = plt.imread("datasetTest/{}/color/aligned_color_crop{}.png".format(snapshot_file,i)) 
                plt.imshow(color_output, vmin=-1, vmax=1, cmap='RdYlGn')
                plt.plot([rect[0][0], rect[1][0]], [rect[0][1], rect[1][1]], linewidth=2, color='blue')
                plt.plot([rect[2][0], rect[3][0]], [rect[2][1], rect[3][1]], linewidth=2, color='blue')
                plt.scatter(preds[i,0], preds[i,1], color='blue')
                plt.scatter(preds[i,2], preds[i,3], color='purple')
                plt.savefig('results/{}/color/{}_color_result.png'.format(snapshot_file,i), dpi=120)  
                plt.clf()
    else:
        print("end of training !")

def signal_handler(sig, frame):
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX signal_handler')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX signal_handler')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX signal_handler')
    global g_node
    #sys.exit(0)

def gripper_action(bit):
    if bit==0 :
        print("ouverture gripper")
        time.sleep(0.2)
    if bit==1 :
        print("fermeture gripper")
        time.sleep(0.2)
    return
    
def pointsToWorld(inv_t_form,imagePoints):
    
    WorldPoint=np.dot(imagePoints,inv_t_form)
    WorldPoint=WorldPoint/WorldPoint[0,2]
    
    return(WorldPoint) 

def start(training,image,viz,cnn,snapshot_file,epoch):
    
    save = False
    grasp = False
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Allows to chose several arguments')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store', default=snapshot_file ,help='Type of object')
    parser.add_argument('--save', dest='save', action='store', default=save ,help='Save images and files during the process')
    parser.add_argument('--viz', dest='viz', action='store', default=viz ,help='Display of the results')
    parser.add_argument('--image', dest='image', action='store', default=image ,help='Working from camera acquisition if set to false, and from npy saved files if set to true')
    parser.add_argument('--training', dest='training', action='store', default=training ,help='Run the training or not')
    parser.add_argument('--grasp', dest='grasp', action='store', default=grasp ,help='Gripper grasps or not')
    parser.add_argument('--cnn', dest='cnn', action='store', default=cnn ,help='Choose the convolution network')
    parser.add_argument('--epoch', dest='epoch', action='store', default=epoch ,help='Number of iterations')
    args = parser.parse_args()
    launch(args)
    return
    
if __name__=="__main__":
    root = tk.Tk()
    root.title('IA Grasping App')
    root.resizable(False,False)

    def launcher():
        train=t.get()
        img=True
        viz=False
        cnn=c.get()

        snapshot_file = snap.get()
        epoch = int(epoc.get())
        start(train,img,viz,cnn,snapshot_file,epoch)
        return
    
          
        
    t = tk.BooleanVar()
    i = tk.BooleanVar()
    v = tk.BooleanVar()
    c = tk.IntVar()
    c.set(0) 
    snap = tk.StringVar(root, value='bottle')
    epoc = tk.StringVar(root, value='4000')
    p = tk.IntVar(value=20)
  
    
    # train= False
    # img= False
    # viz= False
    nb_rows = 0
    
    training = [("Yes", True),("No", False)]
# 
    # networks = [("Resnet", 0), ("Pruned Resnet", 1), ("Densenet", 2), ("VGG19", 3)]
    networks = [("Resnet", 0)]

    tk.Label(root, 
         text="""Train network :""",
         justify = tk.CENTER,padx = 80,pady=6).grid(row=nb_rows,column=0)
    tk.Label(root, 
         text="""Number of iterations:""",
         justify = tk.CENTER,padx = 80,pady=10).grid(row=nb_rows+2,column=1)
    nb_rows=nb_rows+1
    
    for language, val in training:
            tk.Radiobutton(root, 
                  text=language,
                  indicatoron = 0,
                  width = 20,
                  padx = 50, 
                  pady= 8,
                  variable=t, 
                  value=val).grid(row=nb_rows,column=0)
            nb_rows=nb_rows+1      
    
    e=tk.Entry(root, textvariable=epoc).grid(row=nb_rows,column=1)    
    
    
    # tk.Label(root, 
    #      text="""Use saved images:""",
    #      justify = tk.CENTER,padx = 80,pady=10).grid(row=nb_rows,column=0)

    # tk.Label(root, 
    #      text="""Visualization:""",
    #      justify = tk.CENTER,padx = 80,pady=10).grid(row=nb_rows,column=1)
    # nb_rows=nb_rows+1
    
    
    # for language, val in image:
    #         tk.Radiobutton(root, 
    #               text=language,
    #               indicatoron = 0,
    #               width = 20,
    #               padx = 50, 
    #               pady= 8,
    #               variable=i, 
    #               value=val).grid(row=nb_rows,column=0)
    #         nb_rows=nb_rows+1
    #         tmp=tmp+1 
    # nb_rows=nb_rows-tmp       
    # for language, val in visu:
    #         tk.Radiobutton(root, 
    #               text=language,
    #               indicatoron = 0,
    #               width = 20,
    #               padx = 50, 
    #               pady= 8,
    #               variable=v,
    #               value=val).grid(row=nb_rows,column=1)
    #         nb_rows=nb_rows+1

    
    tk.Label(root, 
         text="""Convolution network:""",
         justify = tk.CENTER,padx = 80,pady=10).grid(row=nb_rows,column=0)
    nb_rows=nb_rows+1
    
    for language, val in networks:
            tk.Radiobutton(root, 
                  text=language,
                  indicatoron = 0,
                  width = 20,
                  padx = 50, 
                  pady= 8, 
                  variable=c, 
                  value=val).grid(row=nb_rows,column=0)   
            nb_rows=nb_rows+1
    
    tk.Label(root, 
         text="""Folders name:""",
         justify = tk.CENTER,padx = 80,pady=10).grid(row=nb_rows-5,column=1)
    e=tk.Entry(root, textvariable=snap).grid(row=nb_rows-4,column=1)
    
    # GenButton = tk.Button(root, 
    #                text="Demos images generator", 
    #                width = 20,
    #                padx = 50, 
    #                pady=8,
    #                command=gen)
    # GenButton.grid(row=nb_rows-2,column=1)
    
    # CaptButton = tk.Button(root, 
    #                text="Capture images", 
    #                width = 20,
    #                padx = 50, 
    #                pady=8,
    #                command=capture)
    # CaptButton.grid(row=nb_rows-1,column=1)
    
    tk.Label(root, 
         text="""""",
         justify = tk.CENTER,padx = 20,pady=2).grid(row=nb_rows)
    nb_rows=nb_rows+1
    
    button = tk.Button(root, 
                   text="QUIT", 
                   fg="red",
                   width = 20,
                   pady=8,
                   command=quit)
    button.grid(row=nb_rows,column=0)

    slogan = tk.Button(root,
                   text="Launch",
                   width = 20,
                   pady=8,
                   command=launcher)
    slogan.grid(row=nb_rows,column=1)
    nb_rows=nb_rows+1
    tk.Label(root, 
         text="""""",
         justify = tk.CENTER,padx = 20,pady=2).grid(row=nb_rows)
    root.mainloop()
    