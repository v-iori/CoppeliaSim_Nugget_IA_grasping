import tensorflow as tf

# Available backbones : DenseNet121 and VGG19
class ResNetFeatModel():
    def __init__(self):
        '''
        Resnet is trained on 224x224 pixels images
        '''
        super(ResNetFeatModel, self).__init__()
        baseModel = tf.keras.applications.ResNet50(weights='imagenet')
        # baseModel.summary()
        print("Use of Resnet model")
        self.model = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.get_layer("conv5_block3_add").output)
        for layer in self.model.layers:
            layer.trainable = False

    def __call__(self, inputs):
       output = self.model(inputs)
       return output


# Secondary NN (NN after the backbone)

class BaseDeepModel(tf.keras.Model):
    def __init__(self):
        super(BaseDeepModel, self).__init__()
        pass


class GraspNetTest(BaseDeepModel):
    def __init__(self):
        super(GraspNetTest, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167
        # self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")
        dropout_rate = 0.4
        ### First layer ###
        self.conv0 = tf.keras.layers.Convolution2D(128, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv0", trainable=True)
        self.drop0 = tf.keras.layers.Dropout(dropout_rate)
        ### Second Layer ###
        self.upconv0 = tf.keras.layers.UpSampling2D((2, 2))
        ### Third layer ###
        self.bn1 = tf.keras.layers.BatchNormalization(name="grasp-b1")
        self.conv1 = tf.keras.layers.Convolution2D(256, kernel_size=3, strides=1, activation=tf.nn.relu,
                                                   use_bias=True, padding='same', name="grasp-conv1", trainable=True)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        ### Classification layer ###
        self.bn2 = tf.keras.layers.BatchNormalization(name="grasp-b2")
        self.conv2 = tf.keras.layers.Convolution2D(1, kernel_size=1, strides=1, activation=tf.nn.tanh,
                                                   use_bias=False, padding='same', name="grasp-conv2", trainable=True)
  
    def call(self, inputs, bufferize=False, step_id=-1):
        ### First layer ###
        x = self.conv0(inputs)
        x = self.drop0(x)
         ### Second Layer ###
        x = self.upconv0(x)
        ### Third layer ###
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        ### Classification layer ###
        # x = self.bn2(x)
        x = self.conv2(x)
        return x

# Chosen architecture : backbone + GraspNetTest
class FullNN(tf.keras.Model):
    def __init__(self,cnn=0):
        super(FullNN, self).__init__()
        if (cnn==0):
            self.CNN = ResNetFeatModel()
        self.QGraspTest = GraspNetTest()
        # initialize variables 
        self.in_height, self.in_width = 0, 0
        self.scale_factor = 2.0
        self.padding_width = 0
        self.target_height = 0
        self.target_width = 0
        
    
    def call(self, input):
        x = self.QGraspTest(self.CNN(input))
        return x


