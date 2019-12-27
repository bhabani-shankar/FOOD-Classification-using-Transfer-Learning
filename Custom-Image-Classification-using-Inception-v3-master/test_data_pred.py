import tensorflow as tf
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
__author__ = 'ibininja'
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'F:/Projects/Custom-Image-Classification-using-Inception-v3-master/testing-images')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

        image_path = destination

        image_data = tf.io.gfile.GFile(image_path, 'rb').read()
        label_lines = [line.rstrip() for line
                            in tf.io.gfile.GFile("tf_files/retrained_labels.txt")]

        with tf.io.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            arr=[]
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
                arr.append(human_string)
        return jsonify({"food name": arr[0]})

    return render_template("complete.html")


if __name__ == "__main__":
    app.run(port=4555, debug=True)
