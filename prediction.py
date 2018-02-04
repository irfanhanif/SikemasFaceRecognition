import pickle
import cv2
import sys
import operator
import base64
import json
import numpy as np
import openface
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

print "Loading model"
predictor_model = "shape_predictor_68_face_landmarks.dat"
network_model = "nn4.small2.v1.t7"
face_aligner = openface.AlignDlib(predictor_model)
net = openface.TorchNeuralNet(network_model, 96, cuda='store_true')

model_path = 'kelas/' + sys.argv[1] + '/'
model_file = model_path + 'model.pkl'
model = pickle.load(open(model_file, 'rb'))

print "Ready!"

@app.route('/predict', methods=['POST'])
def ApiCall():
    print "starting..."
    try:
        img = cv2.imdecode(np.fromstring(base64.b64decode(request.values['imagefile']), np.uint8), cv2.IMREAD_COLOR)
        print "image decoded"

        aligned_img = face_aligner.align(96, img, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        print "Image aligned"

        rep = net.forward(aligned_img)
        print "representation found"

        predicted_proba = model.predict_proba([rep])

        res = {}
        for i in range(len(model.classes_)):
            res[model.classes_[i]] = predicted_proba[0][i]
        res = sorted(res.items(), key=operator.itemgetter(1))
        res.reverse()

        rank = 0
        for key, val in res:
            rank += 1
            if key == request.values['nrp']:
                print "PROBABILITY RANK: " + str(rank)
                prob = val
                break

        if rank <= 5:
            return jsonify(status='200', message='ok', validation='accepted', probability=prob)
        else:
            return jsonify(status='200', message='ok', validation='rejected', probability=prob)

    except Exception as err:
        print str(err)
        return jsonify(status='500', message='internal server error')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
