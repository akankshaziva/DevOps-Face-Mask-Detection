from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
application = Flask(__name__)

dic = {0 : 'Mask Wear Ok', 1 : 'Mask Below Nose',2: "Mask On Chin Please wear properly",3: "Please wear mask"}

model = load_model('face_mask.model')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	test = image.img_to_array(i)
	test = np.array(test, dtype="float32")
	test = test / 255.
	test = np.expand_dims(test, axis=0)
	pred = model.predict(test)
	pred = np.argmax(pred, axis=1)
	i=pred[0]
	return dic[i]

# routes
@application.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@application.route("/about")
def about_page():
	return "Msdhoni!!!"

@application.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	application.run(debug = True)