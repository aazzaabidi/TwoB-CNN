import tensorflow as tf
from tensorflow.keras.layers import Layer, Activation, Dense, Conv1D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization, Concatenate, Add
from tensorflow.python.ops.gen_batch_ops import Batch
tf.keras.backend.set_floatx('float32')

class OneBranchModel(tf.keras.Model):
    def __init__(self,n_classes,encoder,regularizer=None):
        super(OneBranchModel, self).__init__(name='OneBranchModel')
        if encoder == '1D':
            self.branch = CNN1D() 
            self.out = tf.keras.Sequential([
                Dense(128,activation='relu'),
                BatchNormalization(),
                Dense(256,activation='relu'),
                Dense(n_classes,activation='softmax')
            ])
        elif encoder == '2D':
            self.branch = CNN2D()
            self.out = Dense(n_classes,activation='softmax')
        elif encoder == '3D':
            self.branch = CNN3D()
            self.out = Dense(n_classes,activation='softmax')
        elif encoder == 'TempCNN':
            self.branch = TempCNN()
            self.out = tf.keras.Sequential([
                Dense(256,kernel_regularizer=tf.keras.regularizers.l2(l=1E-6)),
                BatchNormalization(),
                Activation('relu'),
                Dropout(rate=0.5),
                Dense(n_classes,activation='softmax',kernel_regularizer=regularizer)
            ])
    def call (self,inputs,is_training):
        feat = self.branch(inputs,is_training)
        return self.out(feat)

class TwoBranchModel(tf.keras.Model):
    def __init__(self,n_classes,encoder,fusion='concat',num_units=32):
        super(TwoBranchModel, self).__init__(name='TwoBranchModel')
        if encoder == '1D':
            self.branch1 = CNN1D()
            self.branch2 = CNN1D() 
        elif encoder == 'TempCNN':
            self.branch1 = TempCNN()
            self.branch2 = TempCNN()
        self.fusion = Add()
        self.out = tf.keras.Sequential([
                Dense(256,activation='relu'),
                BatchNormalization(),
                Dense(n_classes,activation='softmax')
        ])
        self.aux1 = Dense(n_classes,activation='softmax')
        self.aux2 = Dense(n_classes,activation='softmax')
    def call (self,inputs1,inputs2, is_training):
        feat1 = self.branch1(inputs1,is_training)
        feat2 = self.branch2(inputs2,is_training)
        feat = self.fusion([feat1,feat2])
        return self.out(feat), self.aux1(feat1), self.aux2(feat2)

class CNN1D (tf.keras.Model):
    def __init__(self, n_filters=128, dropout_rate = 0.4):
        super(CNN1D, self).__init__(name='CNN1D')
        self.conv1 = Conv1D(filters=n_filters, kernel_size=3, padding='valid', name="conv1_", activation="relu")
        self.Dropout1 = Dropout(rate=dropout_rate, name="dropOut1_")
        self.conv2 = Conv1D(filters=n_filters, kernel_size=3, padding='valid', name="conv2_", activation="relu")
        self.Dropout2 = Dropout(rate=dropout_rate, name="dropOut2_")
        self.conv3 = Conv1D(filters=n_filters*2, kernel_size=3, padding='valid', name="conv3_", activation="relu")
        self.Dropout3 = Dropout(rate=dropout_rate, name="dropOut3_")
        self.conv4 = Conv1D(filters=n_filters*2, kernel_size=1, padding='valid', name="conv4_", activation="relu")
        self.Dropout4 = Dropout(rate=dropout_rate,name="dropOut4_")
        self.globPool = GlobalAveragePooling1D()

    def call (self,inputs,is_training):
        x = self.conv1(inputs)
        x = self.Dropout1(x,is_training)
        x = self.conv2(x)
        x = self.Dropout2(x,is_training)
        x = self.conv3(x)
        x = self.Dropout3(x,is_training)
        x = self.conv4(x)
        x = self.Dropout4(x,is_training)
        pool = self.globPool(x)
        return pool

class CNN2D (tf.keras.Model):
    '''
    2D-CNN model from (Ji et al, 2018)
    https://www.mdpi.com/2072-4292/10/1/75/htm
    '''
    def __init__(self,n_filters=32):
        super(CNN2D, self).__init__(name='CNN2D')
        self.conv1 = Conv2D (filters=n_filters, kernel_size=(3,3), padding='same', activation='relu')
        self.pool1 = MaxPooling2D (pool_size=(2,2),padding='same')
        self.conv2 = Conv2D (filters=n_filters, kernel_size=(3,3), padding='same', activation='relu')
        self.pool2 = MaxPooling2D (pool_size=(2,2),padding='same')
        self.conv3 = Conv2D (filters=n_filters*2, kernel_size=(3,3), padding='same', activation='relu')
        self.pool3 = MaxPooling2D (pool_size=(2,2),padding='same')
        self.flatten = Flatten()
    def call (self,inputs,is_training):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return self.flatten(x)

class CNN3D (tf.keras.Model):
    '''
    3D-CNN model from (Ji et al, 2018)
    https://www.mdpi.com/2072-4292/10/1/75/htm
    '''
    def __init__(self,n_filters=32):
        super(CNN3D, self).__init__(name='CNN3D')
        self.conv1 = Conv3D (filters=n_filters, kernel_size=(3,3,3), padding='same', activation='relu')
        self.pool1 = MaxPooling3D (pool_size=(2,2,1),padding='same')
        self.conv2 = Conv3D (filters=n_filters, kernel_size=(3,3,3), padding='same', activation='relu')
        self.pool2 = MaxPooling3D (pool_size=(2,2,1),padding='same')
        self.conv3 = Conv3D (filters=n_filters*2, kernel_size=(3,3,3), padding='same', activation='relu')
        self.pool3 = MaxPooling3D (pool_size=(2,2,1),padding='same')
        self.flatten = Flatten()
    def call (self,inputs,is_training):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return self.flatten(x)

        
class TempCNN (tf.keras.Model):
    '''
    TempCNN encoder from (Pelletier et al, 2019) 
    https://www.mdpi.com/2072-4292/11/5/523
    '''
    def __init__(self,n_filters=64,drop=0.5):
        super(TempCNN, self).__init__(name='TempCNN')
        self.conv1 = Conv1D(filters=n_filters, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(l=1E-6))
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.drop_layer1 = Dropout(rate=drop)
        self.conv2 = Conv1D(filters=n_filters, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(l=1E-6))
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.drop_layer2 = Dropout(rate=drop)
        self.conv3 = Conv1D(filters=n_filters, kernel_size=1, kernel_regularizer=tf.keras.regularizers.l2(l=1E-6))
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')
        self.drop_layer3 = Dropout(rate=drop)
        self.flatten = Flatten()
    def call(self,inputs, is_training):
        x = self.drop_layer1(self.act1(self.bn1(self.conv1(inputs))),is_training)
        x = self.drop_layer2(self.act2(self.bn2(self.conv2(x))),is_training)
        x = self.drop_layer3(self.act3(self.bn3(self.conv3(x))),is_training)
        return self.flatten(x)


