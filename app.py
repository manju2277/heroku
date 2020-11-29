from flask import Flask, render_template, request
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2
app = Flask(__name__)
@app.route("/")
def home():
  return render_template('base.html')
@app.route('/after',methods=['GET','POST'])
def after():
  img=request.files['file1']
  img.save('./static/file.jpg')
  image=cv2.imread('./static/file.jpg')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
  faces = cascade.detectMultiScale(gray, 1.1, 3)

  for x,y,w,h in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cropped=image[y:y+h,x:x+w]
  cv2.imwrite('./static/after.jpg', image)

  try:
    cv2.imwrite('./static/cropped.jpg',cropped)
  except:
    pass
  
  try:
    image = cv2.imread('./static/cropped.jpg', 0)
  except:
    image = cv2.imread('./static/file.jpg', 0)

  image = cv2.resize(image, (48,48))

  image = image/255.0

  image = np.reshape(image, (1,48,48,1))

  #load model
  model = model_from_json(open("model.json", "r").read())
  #load weights
  model.load_weights('modelweights.h5')


  prediction = model.predict(image)

  label_map =   ['Anger','Disgust','Fear', 'Happy', 'Sad', 'Surprise','Neutral']

  prediction = np.argmax(prediction)

  final_prediction = label_map[prediction]

  return render_template('after.html',data=final_prediction)
    
if __name__ == "__main__":
    app.run()
