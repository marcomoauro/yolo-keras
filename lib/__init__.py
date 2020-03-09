from weight_reader import WeightReader
import yolo_model as m
import utils as u
from keras.models import load_model
import boxes_utils as bu
import labels as l

# CONFIGS
weights_filename = 'weights/yolov3.weights'
model_filename = 'models/model.h5'
photo_filename = 'images/220693.jpg'
# define the expected input shape for the model
input_w, input_h = 416, 416
# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
# define the probability threshold for detected objects
class_threshold = 0.6

# define the model
model = m.make_yolov3_model()
# load the model weights
weight_reader = WeightReader(weights_filename)
# set the model weights into the model
weight_reader.load_weights(model)
# save the model to file
model.save(model_filename)
# load yolov3 model
model = load_model(model_filename)
# define our new photo
# load and prepare image
image, image_w, image_h = u.load_image_pixels(photo_filename, (input_w, input_h))
# make prediction
yhat = model.predict(image)
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += u.decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
bu.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
bu.do_nms(boxes, 0.5)
# define the labels
labels = l.labels()
# get the details of the detected objects
v_boxes, v_labels, v_scores = bu.get_boxes(boxes, labels, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
# draw what we found
bu.draw_boxes(photo_filename, v_boxes, v_labels, v_scores)