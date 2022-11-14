from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"

#@app.route("/sum", methods=['POST'])
#def sum():
#    x = request.json['x']
#    y = request.json['y']
#    z = x + y 
#    return {'sum':z}


#Load one of the saved models
model_path = "svm_Gamma=0.01_C=0.5.joblib"
model = load(model_path)
print("Model Loaded Successfully!")

	
@app.route("/predict", methods=['POST'])
def predict_digit():

	image1 = request.json['image1']
	image2 = request.json['image2']
	
	predicted = model.predict([image1])
	predicted_output_1 = int(predicted[0])
	
	predicted = model.predict([image2])
	predicted_output_2 = int(predicted[0])

	if predicted_output_1 == predicted_output_2:
		return '\nImages are Same \n\n'
	else:
		return '\nImages are not Same \n\n'

		
if __name__ == "__main__":
    app.run(debug=True)
