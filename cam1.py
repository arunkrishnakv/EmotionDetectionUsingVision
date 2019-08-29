from keras.models import load_model 
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from pathlib import Path
import numpy as np
import cv2
from flask import Flask,render_template,url_for,request
import random
import sys
import os


video_capture = cv2.VideoCapture(cv2.CAP_DSHOW)
#video_capture = cv2.VideoCaptureAsync(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
#video_capture.start()
#frame=np.empty(4,dtype=int)
#frame = np.expand_dims(frame, axis=0)
frame=[]
frame = np.empty(4,dtype=int)
#video_capture.retrieve(frame);
ret,frame=video_capture.read()
#video_capture.read(frame)
# Close device
video_capture.release()
x=random.randint(1,10000)
pat='C:/Users/arunk/Desktop/Edu/fstival/static/image'+str(x)+'.png'
cv2.imwrite(pat,frame)

cv2.imshow(pat, frame)
#path='C:/Users/Malusha/Desktop/fstival/image.png'
cv2.destroyAllWindows()
#video_capture.stop()

# Load the model we trained
model = load_model('weights1.h5')

# Load an image file to test
#image_to_test = image.load_img(r'C:\Users\Sparrow\Desktop\face\142.jpg',target_size=(64, 64))

# Convert the image data to a numpy array suitable for Keras
image_to_test = frame

# Normalize the image the same way we normalized the training data (divide all numbers by 255)
image_to_test = image_to_test/255

# Add a fourth dimension to the image since Keras expects a list of images
images = np.expand_dims(image_to_test, axis=0)
images = np.resize(images, (-1,64,64,3))
#images = cv2.resize(images,(64,64,3))

# Make a prediction using the bird model
results = model.predict(images)

# Since we only passed in one test image, we can just check the first result directly.
image_likelihood = results[0][0]

print(image_likelihood)

f=0
s='' 
if image_likelihood<0.0009: 
  s='no face detected' 
  f=1  

results=results.tolist()
i=results[0].index(max(results[0]))
print(i)
print(pat)
PEOPLE_FOLDER = 'static'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
pat='image'+str(x)+'.png'
pat = os.path.join(app.config['UPLOAD_FOLDER'], pat)

if f==1:
  @app.route('/')
  
  def xno():
    return render_template("xno.html",value=s,f=image_to_test,path=pat)
  print('No face detected')

if i == 0 and f==0:
  @app.route('/')
  
  def angry():
    return render_template("xangry.html",value=s,f=image_to_test,path=pat)
  #app.run(debug=True)
  print("angry")
if i == 1 and f==0:
  @app.route('/')
  def disgust():
    return render_template("xdisgust.html",value=s,f=image_to_test,path=pat)
  #app.run(debug=True)
  print('disgust')
if i == 2 and f==0:
  @app.route('/')
  def happy():
    return render_template("xhappy.html",value=s,f=image_to_test,path=pat)
  #app.run(debug=True)
  print('happy')
if i == 3 and f==0:
  @app.route('/')
  def neutral():
    return render_template("xneutral.html",value=s,f=image_to_test,path=pat)
  #app.run(debug=True)
  print('neutral')
if i == 4 and f==0:
  @app.route('/')
  def sad():
    return render_template("xsad.html",value=s,f=image_to_test,path=pat)
  #app.run(debug=True)
  print('sad')
if i == 5 and f==0:
  @app.route('/')
  def surprise():
    return render_template("xsurprise.html",value=s,f=image_to_test,path=pat)
  #app.run(debug=True)
  print('surprise')


#@app.route('/')
#def index():
#  return render_template("xresult.html",value=i,f=image_to_test,path=pat)
#app.run(port=80,debug=True)
app.run(debug=True)



