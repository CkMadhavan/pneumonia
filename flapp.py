from flask import Flask , render_template , request

app = Flask(__name__)

import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow import keras

im_size=512

project_name = "Chest Xray Multilabel CNN"
learning_rate = 1e-3
momentum = 0.9
epochs = 1
steps_per_epoch = 1
decay = learning_rate/epochs

os.system("wget https://www.dropbox.com/s/441b2fux12sgktp/chest.h5?dl=1 -O chest.h5")
os.system("wget https://www.dropbox.com/s/kzs3y7fl14an3nd/model.json?dl=1 -O model.json")

inp1 = keras.layers.Input(shape = (im_size,im_size,1))

o = keras.layers.Conv2D(64,(3,3) , padding="same")(inp1)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(64,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(64,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(64,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(128,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(128,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(128,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(128,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(256,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(256,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(256,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(256,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(256,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(256,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(512,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(512,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(512,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(512,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(1024,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(1024,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Conv2D(1024,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Conv2D(1024,(3,3) , padding="same")(o)
o = keras.layers.Activation('relu')(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.MaxPooling2D((2,2),padding='same')(o)

o = keras.layers.Flatten()(o)
o = keras.layers.Dense(512 , activation = 'relu')(o)
o = keras.layers.BatchNormalization()(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Dense(1024,activation = 'relu')(o)
o = keras.layers.BatchNormalization()(o)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.Dense(2048, activation = 'relu')(o)
o = keras.layers.BatchNormalization()(o)
o = keras.models.Model(inputs=inp1, outputs=o)

inp2 = keras.layers.Input(shape = (1,))
t = keras.layers.Dense(512 , activation = 'relu')(inp2)
t = keras.layers.BatchNormalization()(t)
t = keras.layers.Dense(1024 , activation = 'relu')(t)
t = keras.layers.BatchNormalization()(t)
t = keras.layers.Dropout(0.2)(t)
t = keras.models.Model(inputs=inp2, outputs=t)

combined = keras.layers.concatenate([o.output, t.output])

z = keras.layers.Dense(512, activation = 'relu')(combined)
z = keras.layers.BatchNormalization()(z)
z = keras.layers.Dropout(0.2)(z)
z = keras.layers.Dense(1024 , activation = 'relu')(z)
z = keras.layers.BatchNormalization()(z)
z = keras.layers.Dropout(0.2)(z)
z = keras.layers.Dense(2048 , activation = 'relu')(z)
z = keras.layers.BatchNormalization()(z)
z = keras.layers.Dense(14 , activation = 'sigmoid')(z)

model = keras.models.Model(inputs=[o.input, t.input], outputs=z)

model.compile(loss = 'binary_crossentropy' , optimizer = keras.optimizers.SGD(lr=learning_rate , momentum=momentum , decay= decay))

@app.route('/')
def index():
    return render_template('up.html')

@app.route('/check')
def check():
    
    #with open("model.json") as filename:
        #json_file = filename.read()
        #model = keras.models.model_from_json(filename.read())

    model.load_weights("chest.h5")

    return str(model.count_params())

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       
      input_dim = 256
      file_names = []

      f = request.files['file']
      uploaded_files = request.files.getlist("file")

      infer = []

      for f in uploaded_files:

        file_names.append(f.filename)

        f = Image.open(f)
        f = f.resize((input_dim,input_dim))
        f = f.convert('L')
        a = np.asarray(f)
        a = a.reshape(1,input_dim,input_dim,1)
        infer.append(a)
        
      infer = np.array(infer).reshape(-1 , input_dim,input_dim,1)

      m = tf.keras.Sequential()

      m.add(tf.keras.layers.Conv2D(256 ,(3,3) , input_shape = (input_dim , input_dim , 1)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
      m.add(tf.keras.layers.Dropout(0.4))

      m.add(tf.keras.layers.Conv2D(256 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
      m.add(tf.keras.layers.Dropout(0.4))
        
      m.add(tf.keras.layers.Conv2D(256 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
      m.add(tf.keras.layers.Dropout(0.4))
        
      m.add(tf.keras.layers.Conv2D(256 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
      m.add(tf.keras.layers.Dropout(0.4))

      m.add(tf.keras.layers.Flatten())
      m.add(tf.keras.layers.Dense(64))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Dense(1))
      m.add(tf.keras.layers.Activation('sigmoid'))

      m.compile(loss = 'binary_crossentropy' , optimizer = tf.keras.optimizers.Adam(lr = 0.00001) , metrics = ['acc'])

      m.load_weights("weight.h5")

      p = (m.predict(infer)).tolist()

      print(p)
        
      x = []
    
      for i in range(len(p)):
        if p[i][0] > 0.50:
            text = file_names[i] + " : Pneumonia"
            x.append(text)
        else:
            text = file_names[i] + " : Normal"
            x.append(text)


      tf.keras.backend.clear_session()


      return render_template('out.html' , i = x)

@app.route('/help')
def hel():
    return render_template('help.html')    

if __name__ == "__main__":
    app.run()
