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

      m = tf.keras.Sequential()

      m.add(tf.keras.layers.Conv2D(256 ,(3,3) , input_shape = (28 , 28 , 1)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

      m.add(tf.keras.layers.Conv2D(256 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

      m.add(tf.keras.layers.Conv2D(256 ,(3,3)))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

      m.add(tf.keras.layers.Flatten())
      m.add(tf.keras.layers.Dense(64))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Dense(64))
      m.add(tf.keras.layers.Activation('relu'))
      m.add(tf.keras.layers.Dense(1))
      m.add(tf.keras.layers.Activation('sigmoid'))

      m.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['acc'])

      m.load_weights("weight.h5")

      p = (m.predict(a))

      print(p[0][0])

      if p > 0.5:
        x = "Predicted To Have Pneumonia With Confidence " + str(p*100) + "%"
      else:
        x = "Predicted To Not Have Pneumonia With Confidence " + str(100- (p*100.0)) + "%"


      tf.keras.backend.clear_session()


      return x

if __name__ == "__main__":
    app.run()
