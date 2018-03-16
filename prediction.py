import pickle
import cv2
import sys
import operator
import base64
import json
import numpy as np
import openface
import gc
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

print "Loading neural network model nn4.small2.v1.t7 ..."
print "Loading shape_predictor_68_face_landmarks ..."
predictor_model = "shape_predictor_68_face_landmarks.dat"
network_model = "nn4.small2.v1.t7"
face_aligner = openface.AlignDlib(predictor_model)
net = openface.TorchNeuralNet(network_model, 96, cuda='store_true')
print "Neural networks and shape predictor model loaded!"

# Dictionary for loaded model object
model_loaded = {}

@app.route('/loadmodel', methods=['POST'])
def load_model():
    kelas_path = "kelas/"
    kelas_name = request.values['kelas']
    model_name = "model.pkl"

    model_path = kelas_path + kelas_name + '/'
    model_file = model_path + model_name

    if kelas_name in model_loaded:
        message = "Model has already loaded"
        return jsonify(status='200', reply=False, message=message)

    try:
        print "Loading Random Forests model for " + kelas_name + "..."
        f = open(model_file, 'rb')
        model_obj = pickle.load(f)
        model_loaded[kelas_name] = model_obj

        message = "Model " + kelas_name + " loaded!"
        print message
        return jsonify(status='200', reply=True, message=message)
    except Exception as err:
        print str(err)
        message = "Model cannot be loaded. Please contact admin."
        print message
        return jsonify(status='500', reply=False, message=message)

@app.route('/removemodel', methods=['POST'])
def remove_model():
    kelas_name = request.values['kelas']
    if kelas_name not in model_loaded:
        message = "Model "+ kelas_name +" already closed or has not been opened"
        print message
        return jsonify(status='500', reply=False, message=message)
    else:
        del model_loaded[kelas_name]
        model_loaded.pop(kelas_name, None)
        gc.collect()
        message = "Model"+ kelas_name +" successfully dumped"
        print message
        return jsonify(status='200', reply=True, message=message)

@app.route('/predict', methods=['POST'])
def ApiCall():
    nrp = request.values['nrp']
    kelas_name = request.values['kelas']

    if kelas_name not in model_loaded:
        print "Model not loaded. Prediction process terminated."
        return jsonify(status='500', reply=False, message='Model not loaded. Please contact your lecturer')
    else:
        model = model_loaded[kelas_name]

    print "Start prediction process for " + nrp
    try:
        img = cv2.imdecode(np.fromstring(base64.b64decode(request.values['imagefile']), np.uint8), cv2.IMREAD_COLOR)
        print nrp + " image decoded"

        aligned_img = face_aligner.align(96, img, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        print nrp + " image aligned"

        rep = net.forward(aligned_img)
        print nrp + " representation found"

        predicted_proba = model.predict_proba([rep])

        res = {}
        for i in range(len(model.classes_)):
            res[model.classes_[i]] = predicted_proba[0][i]
        res = sorted(res.items(), key=operator.itemgetter(1))
        res.reverse()

        rank = 0
    	prob = -1
        for key, val in res:
            rank += 1
            if key == request.values['nrp']:
                prob = val
                break

    	if prob == -1:
            print "ERR: Face data of " + nrp + " are currently not registered (500)"
            return jsonify(status='500', reply=False, message='Wajah Anda belum terdaftar di dalam sistem')

        if rank <= 5:
    	    print "Prediction result for " + nrp + " accepted"
            print "Confidence level: " + str(round(prob*100, 2)) + "%"
            print "Probability rank: " + str(rank)
            return jsonify(status='200', message='ok', validation='accepted', probability=prob)
        else:
    	    print "Prediction result for " + nrp + " rejected"
            print "Confidence level: " + str(round(prob*100, 2)) + "%"
            print "Probability rank: " + str(rank)
            return jsonify(status='200', message='ok', validation='rejected', probability=prob)

    except Exception as err:
        return str(err)
        return jsonify(status='500', message='internal server error')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
