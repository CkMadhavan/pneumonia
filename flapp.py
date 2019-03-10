from flask import Flask , render_template , request

app = Flask(__name__)

import tensorflow as tf
from PIL import Image
import numpy as np

@app.route('/')
def index():
    return render_template('up.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':

      f = request.files['file']

      f = Image.open(request.files['file'])
      f = f.resize((28,28))
      f = f.convert('L')
      a = np.asarray(f)
      a = a.reshape(1,28,28,1)

      print(a.shape)

      model = tf.keras.Sequential()

      m.add(tf.keras.layers.Conv2D(512 ,(3,3) , input_shape=(28,28 , 1)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

      m.add(tf.keras.layers.Conv2D(512 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

      m.add(tf.keras.layers.Conv2D(512 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

      m.add(tf.keras.layers.Flatten())
      m.add(tf.keras.layers.Dense(128))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Dense(128))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Dense(1))
      m.add(tf.keras.layers.Activation('sigmoid'))

      m.compile(loss = 'binary_crossentropy' , optimizer = tf.keras.optimizers.Adam(lr = 0.000001) , metrics = ['acc'])

      m.load_weights("weight.h5")

      p = (m.predict(a))

      print(p[0][0])

      if p > 0.5:
        x = "Oh No , This X-Ray Is Predicted To Have Pneumonia"
      else:
        x = "Hurray , This X-Ray Is Predicted To Not Have Pneumonia"


      tf.keras.backend.clear_session()


      return render_template('out.html' , i = x)

@app.route('/help')
def hel():
    return render_template('help.html')    

if __name__ == "__main__":
    app.run()
