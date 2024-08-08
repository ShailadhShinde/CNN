

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pub.execute_input":"2024-06-16T18:45:26.385679Z","iopub.status.idle":"2024-06-16T18:45:26.397311Z","shell.execute_reply.started":"2024-06-16T18:45:26.385647Z","shell.execute_reply":"2024-06-16T18:45:26.396352Z"}}

print("ok") 

pip install py7zr


import py7zr
with py7zr.SevenZipFile('/kaggle/input/statoil-iceberg-classifier-challenge/sample_submission.csv.7z', mode='r') as z:
    z.extractall()
with py7zr.SevenZipFile('/kaggle/input/statoil-iceberg-classifier-challenge/test.json.7z', mode='r') as z:
    z.extractall()
with py7zr.SevenZipFile('/kaggle/input/statoil-iceberg-classifier-challenge/train.json.7z', mode='r') as z:
    z.extractall()

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,ReduceLROnPlateau
import tensorflow as tf
import xgboost as xgb
tf.config.experimental.list_physical_devices()


# %% [code]
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import layers, models
from keras.layers import GlobalMaxPooling2D,Dense,Conv2D,BatchNormalization,MaxPool2D
from keras.layers import MaxPooling2D,Dropout,Flatten,Input,Activation,AvgPool2D,Concatenate,concatenate
import tensorflow
from keras.models import Model,Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score, KFold


train=pd.read_json("/kaggle/working/data/processed/train.json")
test=pd.read_json("/kaggle/working/data/processed/test.json")
testp=pd.read_json("/kaggle/working/data/processed/test.json")
y=train['is_iceberg']

train["inc_angle"]=pd.to_numeric(train["inc_angle"],errors='coerce')
train['inc_angle']=train['inc_angle'].fillna(0)
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')


# %% [code]
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75,75)for band in train['band_1']])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

X_train=np.concatenate([X_band_1[:,:,:,np.newaxis],
                       X_band_2[:,:,:,np.newaxis],
                       ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]],axis=-1)



# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.131112Z","iopub.status.idle":"2024-06-15T22:53:09.131478Z","shell.execute_reply.started":"2024-06-15T22:53:09.131304Z","shell.execute_reply":"2024-06-15T22:53:09.131319Z"}}
X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.134463Z","iopub.status.idle":"2024-06-15T22:53:09.134791Z","shell.execute_reply.started":"2024-06-15T22:53:09.134629Z","shell.execute_reply":"2024-06-15T22:53:09.134643Z"}}
gen=ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0,
                         rotation_range = 0)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.149995Z","iopub.status.idle":"2024-06-15T22:53:09.150329Z","shell.execute_reply.started":"2024-06-15T22:53:09.150147Z","shell.execute_reply":"2024-06-15T22:53:09.150160Z"}}
def Vgg16Model(input_shape=(75, 75, 3),input_meta=1):
    
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape,classes=1)
    for layer in base_model.layers[:-9]:
        layer.trainable = False
    x=base_model.get_layer('block5_pool').output
    
    Global=GlobalMaxPooling2D()(x)

    input_meta = Input(shape=[input_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    concat = concatenate([Global, input_meta_norm], name='features_layer')
   
    dense_layer1=Dense(512,activation='relu')(concat)
    dense_layer2=Dense(256,activation='relu')(dense_layer1)

    predictions=Dense(1,activation='sigmoid')(dense_layer2)

    model = Model(inputs=[base_model.input,input_meta], outputs=predictions)

    opt = SGD(learning_rate=0.001)

    model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy'])
    #model.summary()
    return model
    
    
    
def defined():
    
    model=Sequential([
        
        Input(shape=X_train.shape[1:]),
        
        Conv2D(64,(3,3),activation='relu'),
       # Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(128,(3,3),activation='relu'),
      #  Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
              
        Flatten(),
        
        Dense(256,activation='relu'),
        Dense(128,activation='relu'),
        
        Dense(1,activation='sigmoid')
        
    ])
    
    opt=SGD(learning_rate=0.001)
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    model.summary()
    
    return model

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.160690Z","iopub.status.idle":"2024-06-15T22:53:09.161117Z","shell.execute_reply.started":"2024-06-15T22:53:09.160897Z","shell.execute_reply":"2024-06-15T22:53:09.160916Z"}}


# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.167249Z","iopub.status.idle":"2024-06-15T22:53:09.167569Z","shell.execute_reply.started":"2024-06-15T22:53:09.167408Z","shell.execute_reply":"2024-06-15T22:53:09.167421Z"}}
def get_callbacks(filepath,patience=10):
    es=EarlyStopping('val_loss',patience=patience,mode='min')
    modelsave=ModelCheckpoint(filepath,save_best_only=True,monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    return [es,modelsave,reduce_lr_loss]


def gen_flow_for_two_inputs(X1,X2,y,batch_size):

    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)

    while True:
        X1i = next(genX1)
        X2i = next(genX2)
           
        yield (X1i[0],X2i[1]),X1i[1] 
        

def gen_flow_for_one_input(X1,y,batch_size):
    
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
   
    while True:
        X1i = next(genX1)
        yield X1i[0],X1i[1] 



# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.180962Z","iopub.status.idle":"2024-06-15T22:53:09.181306Z","shell.execute_reply.started":"2024-06-15T22:53:09.181121Z","shell.execute_reply":"2024-06-15T22:53:09.181135Z"}}
num_folds = 3
j=0
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
pred=[]
batch_size=32
for train_index,test_index in kf.split(X_train,y):
    x_train,x_test,traininc,testinc,= X_train[train_index], X_train[test_index],train.iloc[train_index],train.iloc[test_index]
    y_train,y_test = y[train_index], y[test_index]
    j+=1
    
    file_path = "%hhhh.keras"%j
    callbacks = get_callbacks(filepath=file_path, patience=10)
    model=Vgg16Model(input_shape=(75, 75, 3),input_meta=traininc.shape[1])

    model.fit(
       gen_flow_for_two_inputs(x_train,traininc,y_train,batch_size=batch_size),
       steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
       batch_size=32,
       epochs=150,
       verbose=1,
       validation_data=gen_flow_for_two_inputs(x_test,testinc, y_test,batch_size=32),
       validation_steps=int(np.ceil(float(len(x_test)) / float(batch_size))),
       callbacks=callbacks
         )

    model.load_weights(filepath=file_path)
    predi=model8.predict([X_test,test])
    pred.append(predi)



# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.183609Z","iopub.status.idle":"2024-06-15T22:53:09.183921Z","shell.execute_reply.started":"2024-06-15T22:53:09.183769Z","shell.execute_reply":"2024-06-15T22:53:09.183782Z"}}
num_folds = 3
j=0
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
pred=[]
batch_size=32
for train_index,test_index in kf.split(X_train,y):

    x_train,x_test,traininc,testinc,= X_train[train_index], X_train[test_index],np.array(train['inc_angle'][train_index]),np.array(train['inc_angle'][test_index])
    y_train,y_test = y[train_index], y[test_index]
    j+=1
    x_train=np.asarray(x_train).astype(np.float32)
    x_test=np.asarray(x_test).astype(np.float32)
    traininc=np.asarray(traininc).astype(np.float32)
    testinc=np.asarray(testinc).astype(np.float32)

    file_path = "%s_pode_weights.hdf5.keras"%j
    callbacks = get_callbacks(filepath=file_path, patience=10)
    model=Vgg16Model(input_shape=(75, 75, 3),input_meta=1)

    model.fit(
        gen_flow_for_two_inputs(x_train,traininc,y_train,batch_size=batch_size),
        steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
        batch_size=32,
        epochs=150,
        verbose=1,
        validation_data=gen_flow_for_two_inputs(x_test,testinc, y_test,batch_size=32),
        validation_steps=int(np.ceil(float(len(x_test)) / float(batch_size))),
        callbacks=callbacks
         )

    model.load_weights(filepath=file_path)
    predi=model8.predict([X_test,test['inc_angle']])
    pred.append(predi)



# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.186769Z","iopub.status.idle":"2024-06-15T22:53:09.187090Z","shell.execute_reply.started":"2024-06-15T22:53:09.186931Z","shell.execute_reply":"2024-06-15T22:53:09.186945Z"}}
mean=0
for i in range(0,len(pred)):
    mean=mean+pred[i]
    
mean=mean/len(pred)
predicted_test=mean

submission = pd.DataFrame()
submission['id']=testp['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('u24.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.188635Z","iopub.status.idle":"2024-06-15T22:53:09.189065Z","shell.execute_reply.started":"2024-06-15T22:53:09.188838Z","shell.execute_reply":"2024-06-15T22:53:09.188856Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.190266Z","iopub.status.idle":"2024-06-15T22:53:09.190706Z","shell.execute_reply.started":"2024-06-15T22:53:09.190476Z","shell.execute_reply":"2024-06-15T22:53:09.190495Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.191788Z","iopub.status.idle":"2024-06-15T22:53:09.192233Z","shell.execute_reply.started":"2024-06-15T22:53:09.191990Z","shell.execute_reply":"2024-06-15T22:53:09.192008Z"}}
#Import Keras.
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.225791Z","iopub.status.idle":"2024-06-15T22:53:09.226105Z","shell.execute_reply.started":"2024-06-15T22:53:09.225947Z","shell.execute_reply":"2024-06-15T22:53:09.225961Z"}}
from keras.layers import GlobalMaxPooling2D


# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.227169Z","iopub.status.idle":"2024-06-15T22:53:09.227501Z","shell.execute_reply.started":"2024-06-15T22:53:09.227346Z","shell.execute_reply":"2024-06-15T22:53:09.227359Z"}}
def get_model1():
    input1 = layers.Input(shape=(75, 75, 3), name='Data1')

    db1 = layers.BatchNormalization(momentum=0.0)(input1)
    db1 = layers.Conv2D(32, (7,7), activation='relu', padding='same')(db1)
    db1 = layers.MaxPooling2D((2, 2))(db1)
    db1 = layers.Dropout(0.2)(db1)
    
    db2 = layers.Conv2D(64, (5,5), activation='relu', padding='same')(db1)
    db2 = layers.MaxPooling2D((2, 2))(db2)
    db2 = layers.Dropout(0.2)(db2)
    
    db3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(db2)
    db3 = layers.MaxPooling2D((2, 2))(db3)
    db3 = layers.Dropout(0.2)(db3)
    db3 = layers.Flatten()(db3)

    fb1 = layers.Dense(128, activation='relu')(db3)
    fb1 = layers.Dropout(0.5)(fb1)
    output = layers.Dense(1, activation='sigmoid')(fb1)
    
    model = models.Model(inputs=[input1], outputs=[output])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.229843Z","iopub.status.idle":"2024-06-15T22:53:09.230163Z","shell.execute_reply.started":"2024-06-15T22:53:09.230002Z","shell.execute_reply":"2024-06-15T22:53:09.230017Z"}}
def Vgg16Model_single(input_shape=(75, 75, 3)):
    
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape,classes=1)
    for layer in base_model.layers[:-9]:
        layer.trainable = False
   # x=base_model.get_layer('block5_pool').output
    
  #  Global=GlobalMaxPooling2D()(x)

    Global=Flatten()(base_model.output)
    dense_layer1=Dense(256,activation='relu')(Global)
    dense_layer2=Dense(64,activation='relu')(dense_layer1)
#dense_layer2=Dense(64,activation='relu')(dense_layer2)
  #  dense_layer2=Dense(8,activation='relu')(dense_layer2)


    predictions=Dense(1,activation='sigmoid')(dense_layer2)
  
    model = Model(inputs=base_model.input, outputs=predictions)
 
    opt = SGD(learning_rate=0.001)

    model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy'])
    #model.summary()
    return model
    
    

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.231757Z","iopub.status.idle":"2024-06-15T22:53:09.232111Z","shell.execute_reply.started":"2024-06-15T22:53:09.231938Z","shell.execute_reply":"2024-06-15T22:53:09.231953Z"}}
num_folds = 3
j=0
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
pred=[]
input_shape=(75, 75, 3)
batch_size=32

        
for train_index,test_index in kf.split(X_train,y):
    x_train,x_test = X_train[train_index], X_train[test_index]
    y_train,y_test = y[train_index], y[test_index]
    j+=1
    model=Vgg16Model_single()
    file_path = "%uio.keras"%j
    callbacks = get_callbacks(filepath=file_path, patience=10)

    model.fit(
        gen_flow_for_one_input(x_train,y_train,batch_size=batch_size),
        steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
        batch_size=32,
        epochs=150,
        verbose=1,
        validation_data=gen_flow_for_one_input(x_test, y_test,batch_size=32),
        validation_steps=int(np.ceil(float(len(x_test)) / float(batch_size))),
        callbacks=callbacks
         )

    model.load_weights(filepath=file_path)
    predi=model.predict(X_test)
    pred.append(predi)


# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.235850Z","iopub.status.idle":"2024-06-15T22:53:09.236169Z","shell.execute_reply.started":"2024-06-15T22:53:09.236010Z","shell.execute_reply":"2024-06-15T22:53:09.236024Z"}}
mean=0
for i in range(0,len(pred)):
    mean=mean+pred[i]
    
mean=mean/len(pred)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.237285Z","iopub.status.idle":"2024-06-15T22:53:09.237594Z","shell.execute_reply.started":"2024-06-15T22:53:09.237434Z","shell.execute_reply":"2024-06-15T22:53:09.237448Z"}}
predicted_test=mean

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.239003Z","iopub.status.idle":"2024-06-15T22:53:09.239336Z","shell.execute_reply.started":"2024-06-15T22:53:09.239152Z","shell.execute_reply":"2024-06-15T22:53:09.239164Z"}}
submission = pd.DataFrame()
submission['id']=testp['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('pkkk.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.240256Z","iopub.status.idle":"2024-06-15T22:53:09.240592Z","shell.execute_reply.started":"2024-06-15T22:53:09.240419Z","shell.execute_reply":"2024-06-15T22:53:09.240433Z"}}
k

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.242103Z","iopub.status.idle":"2024-06-15T22:53:09.242466Z","shell.execute_reply.started":"2024-06-15T22:53:09.242291Z","shell.execute_reply":"2024-06-15T22:53:09.242306Z"}}
def Vgg16Model_double(input_shape=(75, 75, 3),input_meta=1):
    
    
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape,classes=1)
  #  base_model.trainable = True## Not trainable weights
    for layer in base_model.layers[:-9]:
        layer.trainable = False
    x=base_model.get_layer('block5_pool').output
    
    Global=GlobalMaxPooling2D()(x)

    input_meta = Input(shape=[input_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)


    concat = concatenate([Global, input_meta_norm], name='features_layer')

    dense_layer1=Dense(1024,activation='relu')(concat)
    dense_layer2=Dense(128,activation='relu')(dense_layer1)

    predictions=Dense(1,activation='sigmoid')(dense_layer2)

    model = Model(inputs=[base_model.input,input_meta], outputs=predictions)
    
    opt = SGD(learning_rate=0.001)

    model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy'])
    model.summary()
    return model
    
    

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.266433Z","iopub.status.idle":"2024-06-15T22:53:09.266761Z","shell.execute_reply.started":"2024-06-15T22:53:09.266595Z","shell.execute_reply":"2024-06-15T22:53:09.266608Z"}}
num_folds = 3
j=0
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
pred=[]
batch_size=32
for train_index,test_index in kf.split(X_train,y):
    #print("Train:",train_index,'Test:',test_index)
    x_train,x_test,traininc,testinc,= X_train[train_index], X_train[test_index],train.iloc[train_index],train.iloc[test_index]
    y_train,y_test = y[train_index], y[test_index]
    j+=1
   # x_train=np.asarray(x_train).astype(np.float32)
  #  x_test=np.asarray(x_test).astype(np.float32)
    #traininc=np.asarray(traininc).astype(np.float32)
   # testinc=np.asarray(testinc).astype(np.float32)
    #for i in range(0,10):
 #   j+=1
    file_path = "%pqrst.keras"%j
    callbacks = get_callbacks(filepath=file_path, patience=10)
    model=Vgg16Model_double(input_shape=(75, 75, 3),input_meta=traininc.shape[1])

    model.fit(
       gen_flow_for_two_inputs(x_train,traininc,y_train,batch_size=batch_size),
       steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
       batch_size=32,
       epochs=150,
       verbose=1,
       validation_data=gen_flow_for_two_inputs(x_test,testinc, y_test,batch_size=32),
       validation_steps=int(np.ceil(float(len(x_test)) / float(batch_size))),
       callbacks=callbacks
         )

    model.load_weights(filepath=file_path)
    predi=model.predict([X_test,test])
    pred.append(predi)



# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.267881Z","iopub.status.idle":"2024-06-15T22:53:09.268222Z","shell.execute_reply.started":"2024-06-15T22:53:09.268040Z","shell.execute_reply":"2024-06-15T22:53:09.268054Z"}}

for i in range(0,len(pred)):
    mean=mean+pred[i]
    
mean=mean/len(pred)
predicted_test=mean

submission = pd.DataFrame()
submission['id']=testp['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('uuu3.csv', index=False)


# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.269308Z","iopub.status.idle":"2024-06-15T22:53:09.269631Z","shell.execute_reply.started":"2024-06-15T22:53:09.269472Z","shell.execute_reply":"2024-06-15T22:53:09.269486Z"}}
   
def get_model():
    model=Sequential()
    
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
   
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.271096Z","iopub.status.idle":"2024-06-15T22:53:09.271467Z","shell.execute_reply.started":"2024-06-15T22:53:09.271293Z","shell.execute_reply":"2024-06-15T22:53:09.271307Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.272508Z","iopub.status.idle":"2024-06-15T22:53:09.272830Z","shell.execute_reply.started":"2024-06-15T22:53:09.272667Z","shell.execute_reply":"2024-06-15T22:53:09.272680Z"}}
num_folds = 10
j=0
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
pred=[]
batch_size=32

        
for train_index,test_index in kf.split(X_train,y):
    x_train,x_test = X_train[train_index], X_train[test_index]
    y_train,y_test = y[train_index], y[test_index]
    j+=1
   
    model=get_model()
    file_path = "%file'fllfi.keras"%j
    callbacks = get_callbacks(filepath=file_path, patience=10)

    model.fit(
       gen_flow_for_one_input(x_train,y_train,batch_size=batch_size),
       steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
       batch_size=32,
       epochs=30,
       verbose=1,
       validation_data=gen_flow_for_one_input(x_test, y_test,batch_size=32),
       validation_steps=int(np.ceil(float(len(x_test)) / float(batch_size))),
       callbacks=callbacks
         )

    model.load_weights(filepath=file_path)

    predi=model.predict(X_test)

    pred.append(predi)
    submission = pd.DataFrame()
    submission['id']=testp['id']
    submission['is_iceberg']=predi.reshape((predi.shape[0]))
    submission.to_csv('Ssub'+str(j)+'.csv', index=False)



# %% [code]


# %% [markdown]
# ### Start of the final code

# %% [code] {"execution":{"iopub.status.busy":"2024-06-16T18:46:38.499004Z","iopub.execute_input":"2024-06-16T18:46:38.499597Z","iopub.status.idle":"2024-06-16T18:46:38.644529Z","shell.execute_reply.started":"2024-06-16T18:46:38.499563Z","shell.execute_reply":"2024-06-16T18:46:38.643571Z"}}
import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
seed = 1234
np.random.seed(seed) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Kfold
from sklearn.model_selection import StratifiedKFold

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization
from keras.optimizers import Adam

def get_scaled_imgs(df):

    imgs = []
    
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)    

# %% [code] {"execution":{"iopub.status.busy":"2024-06-16T18:46:38.647264Z","iopub.execute_input":"2024-06-16T18:46:38.647573Z","iopub.status.idle":"2024-06-16T18:46:38.655577Z","shell.execute_reply.started":"2024-06-16T18:46:38.647548Z","shell.execute_reply":"2024-06-16T18:46:38.654636Z"}}
def get_more_images(imgs):

    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images



# %% [code] {"execution":{"iopub.status.busy":"2024-06-16T18:47:20.462170Z","iopub.execute_input":"2024-06-16T18:47:20.462815Z","iopub.status.idle":"2024-06-16T18:47:25.275971Z","shell.execute_reply.started":"2024-06-16T18:47:20.462785Z","shell.execute_reply":"2024-06-16T18:47:25.274892Z"}}
# Training Data
df_train = pd.read_json('/kaggle/working/data/processed/train.json') # this is a dataframe
i=0

Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

# %% [code] {"execution":{"iopub.status.busy":"2024-06-15T22:53:09.288008Z","iopub.status.idle":"2024-06-15T22:53:09.288466Z","shell.execute_reply.started":"2024-06-15T22:53:09.288230Z","shell.execute_reply":"2024-06-15T22:53:09.288249Z"}}
# K fold CV training

i=0
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print("FOLD nr: ", fold_n)
    model = get_model()
    
    MODEL_FILE = 'mdjfjkisllkslfe_k{}_wght.hdf5.keras'.format(fold_n)
    batch_size = 32
    mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')

    model.fit(
        Xtr_more[train], Ytr_more[train],
        batch_size=batch_size,
        epochs=30,
        verbose=1,
        validation_data=(Xtr_more[test], Ytr_more[test]),
        callbacks=[mcp_save, reduce_lr_loss])
    
    model.load_weights(filepath = MODEL_FILE)

    df_test = pd.read_json('/kaggle/working/data/processed/test.json')
    df_test.inc_angle = df_test.inc_angle.replace('na',0)
    Xtest = (get_scaled_imgs(df_test))
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    i=i+1
    submission.to_csv("qmN"+str(i)+".csv", index=False)
    print("submission saved")
wdir = '/kaggle/working/'
stacked_1 = pd.read_csv(wdir + 'qmN1.csv')
stacked_2 = pd.read_csv(wdir + 'qmN2.csv')
stacked_3 = pd.read_csv(wdir + 'qmN3.csv')
stacked_4 = pd.read_csv(wdir + 'qmN4.csv')
stacked_5 = pd.read_csv(wdir + 'qmN5.csv')
stacked_6 = pd.read_csv(wdir + 'qmN6.csv')
stacked_7 = pd.read_csv(wdir + 'qmN7.csv')
stacked_8 = pd.read_csv(wdir + 'qmN8.csv')
stacked_9 = pd.read_csv(wdir + 'qmN9.csv')
stacked_10 = pd.read_csv(wdir + 'qmN10.csv')
sub = pd.DataFrame()
sub['id'] = stacked_1['id']
sub['is_iceberg'] = np.exp(np.mean(
    [
        stacked_1['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_2['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_3['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_4['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_5['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_6['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_7['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_8['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_9['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_10['is_iceberg'].apply(lambda x: np.log(x)),
        ], axis=0))

sub.to_csv(wdir + 'bbbbbbbbbbb.csv', index=False, float_format='%.6f')    

# %% [code]
