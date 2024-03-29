from flask import Flask , render_template , request

app = Flask(__name__)

import tensorflow as tf
from PIL import Image
import numpy as np
import os

@app.route('/')
def index():
    return render_template('up.html')

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
