import io
import os
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

# Imports the Google Cloud client library
from google.cloud import vision

coco_dataset = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5, "bus": 6,
"train": 7, "truck": 8, "boat": 9, "traffic light": 10, "fire hydrant": 11, "street sign": 12, 
"stop sign": 13, "parking meter": 14, "bench": 15, "bird": 16, "cat": 17, "dog": 18, "horse": 19, 
"sheep": 20, "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "hat": 26, "backpack": 27, 
"umbrella": 28, "shoe": 29, "eye glasses": 30, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34, 
"skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38, "baseball bat": 39, "baseball glove": 40, 
"skateboard": 41, "surfboard": 42, "tennis racket": 43, "bottle": 44, "plate": 45, "wine glass": 46, 
"cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51, "banana": 52, "apple": 53, "sandwich": 54, 
"orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61, 
"chair": 62, "couch": 63, "potted plant": 64, "bed": 65, "mirror": 66, "dining table": 67, "window": 68, 
"desk": 69, "toilet": 70, "door": 71, "tv": 72, "laptop": 73, "mouse": 74, "remote": 75, "keyboard": 76, 
"cell phone": 77, "microwave": 78, "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82, 
"blender": 83, "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89, 
"toothbrush": 90, "hair brush": 91}

category_index = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "street sign", 
13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 
20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 26: "hat", 27: "backpack", 
28: "umbrella", 29: "shoe", 30: "eye glasses", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 
35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 
41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 45: "plate", 46: "wine glass", 
47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 
55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 
62: "cake", 63: "couch", 64: "potted plant", 65: "bed", 66: "mirror", 67: "dining table", 68: "window", 
69: "desk", 70: "toilet", 71: "door", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 
77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 
83: "blender", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 
90: "toothbrush", 91: "hair brush"}


def label_to_int(label):
	if label.lower() in coco_dataset:
		return coco_dataset[label.lower()]

	return -1


def create_box(normalized_vertices, w, h):
	box = []
	for vertex in normalized_vertices:
		box.append([vertex.x * w, vertex.y * h])
	return box


# Handles objects
def objects(objects_found, w, h):
	detections_labels = []
	detections_center = []
	detections_scores = []
	detections_boxes = []

	# Iterate over all objects found
	for object_ in objects_found:
		# print('\n                {} (confidence: {})'.format(object_.name, object_.score))
		# print('          Bounding polygon vertices: ')

		centerx = 0
		centery = 0
		count = 0
		for vertex in object_.bounding_poly.normalized_vertices:
			# print('               - ({}, {})'.format(vertex.x * w, vertex.y * h))
			if count == 0 or count == 2:
				centerx += (vertex.x * w)
				centery += (vertex.y * h)
			count += 1

		# print("\n          Center point: ({}, {})".format(centerx / 2.0, centery / 2.0))

		# print(object_.name)
		label_int = label_to_int(object_.name)
		# print(label_int)
		if label_int != -1:
			detections_labels.append(label_int)
			detections_center.append([centerx / 2.0, centery / 2.0])
			detections_scores.append(object_.score)
			detections_boxes.append(create_box(object_.bounding_poly.normalized_vertices, w, h))

	return detections_labels, detections_center, detections_scores, detections_boxes


# Handles frames
def frame(curr_frame, frame_count, scene_truth):
	frame_dict = {"camera": None, "lookat": None, "detections": None, "ground_truth": None}
	detections_dict = {"labels": [], "center": [], "scores": [], "boxes": []}

	# print("       frame #" + str(frame_count))
	frame_dict["camera"] = curr_frame["camera"]
	frame_dict["lookat"] = curr_frame["lookat"]
	frame_dict["ground_truth"] = curr_frame["ground_truth"]

	# Loads image
	content = base64.b64decode(curr_frame["image"])
	image = vision.Image(content=content)
	pil_img = Image.open(io.BytesIO(content))
	w, h = pil_img.size

	# Performs object detection on the image file
	objects_found = client.object_localization(image=image).localized_object_annotations

	# print('\n             Number of objects found: {}'.format(len(objects_found)))

	detections_labels, detections_center, detections_scores, detections_boxes = objects(objects_found, w, h)

	detections_dict["labels"] = detections_labels
	detections_dict["center"] = detections_center
	detections_dict["scores"] = detections_scores
	detections_dict["boxes"] = detections_boxes

	frame_dict["detections"] = detections_dict

	return frame_dict


# Handles scenes
def scene(curr_scene, scene_count):
	scene_dict = {"views": [], "ground_truth": curr_scene["ground_truth"]}

	# print("SCENE #" + str(scene_count))

	frame_count = 0

	# Iterate over all frames in the given scene
	for curr_frame in curr_scene["views"]:
		frame_dict = frame(curr_frame, frame_count, scene_dict["ground_truth"])
		scene_dict["views"].append(frame_dict)
		# print("\n")
		frame_count += 1

	return scene_dict


if __name__ == "__main__":
	# Instantiates a client
	client = vision.ImageAnnotatorClient()

	# Load json
	data = json.load(open("/gpfs/loomis/project/jara-ettinger/mb2987/metagen_project_shared_space/Dataset/75frames_with2d.json", "rb"))

	# Output dict
	output = []

	scene_count = 0

	# print("ALL SCENES")
	# print("_____")
	# print("")

	# Iterate over all scenes
	for curr_scene in data:

		scene_dict = scene(curr_scene, scene_count)

		output.append(scene_dict)

		scene_count += 1

	with open('detections.json', 'w') as outfile:
		json.dump(output, outfile)
	exit()
