import io
import os
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

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

# Returns the object's position on the 2D image in pixel space. (0,0) is the center of the image
def get_image_xy(camera_params, params, obj):
    if on_right_side(camera_params, obj) > 0:
        x, y = locate(camera_params, params, obj)
    else: # on wrong side of camera
        x = math.inf
        y = math.inf

    return x, y


# Check if on the right side of camera. compare to plane going through camera's
# location with a normal vector going from camera to focus
# calculating a(x-x_0)+b(y-y_0)+c(z-z_0).
# if it turns out positive, it's on the focus' side of the camera
def on_right_side(camera_params, obj):
    c = camera_params["camera"]
    f = camera_params["lookat"]
    focus = (f[0] - c[0])*(obj[0] - c[0]) + (f[1] - c[1])*(obj[1] - c[1]) + (f[2] - c[2])*(obj[2] - c[2])
    return focus >= 0


# Given camera info and object's location, find object's location on 2-D image
def locate(camera_params, params, obj):
    a_vertical, b_vertical, c_vertical = get_vertical_plane(camera_params)
    a_horizontal, b_horizontal, c_horizontal = get_horizontal_plane(camera_params, a_vertical, b_vertical, c_vertical)

    s_x = obj[0] - camera_params["camera"][0]
    s_y = obj[1] - camera_params["camera"][1]
    s_z = obj[2] - camera_params["camera"][2]

    s_x_v, s_y_v, s_z_v = proj_vec_to_plane(a_vertical, b_vertical, c_vertical, s_x, s_y, s_z)
    s_x_h, s_y_h, s_z_h = proj_vec_to_plane(a_horizontal, b_horizontal, c_horizontal, s_x, s_y, s_z)

    angle_from_vertical = get_angle(a_vertical, b_vertical, c_vertical, s_x_h, s_y_h, s_z_h)
    angle_from_horizontal = get_angle(a_horizontal, b_horizontal, c_horizontal, s_x_v, s_y_v, s_z_v)
    
    x = params["image_dim_x"] / 2 * (angle_from_vertical / math.radians(params["horizontal_FoV"] / 2))
    y = params["image_dim_y"] / 2 * (-angle_from_horizontal / math.radians(params["vertical_FoV"] / 2))

    x = x + params["image_dim_x"] / 2
    y = y + params["image_dim_y"] / 2

    return x, y


# Returns the angle between the vector and the plane
# a,b,c is coefficients of plane. x, y, z is the vector.
def get_angle(a, b, c, x, y, z):
    numerator = a*x + b*y + c*z # took out abs
    denominator = math.sqrt(math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2)) * math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))
    return math.asin(numerator / denominator) #angle in radians


# Returns a,b,c for the vertical plane. only need the camera parameters
def get_vertical_plane(camera_params):
    p1 = camera_params["camera"]
    p2 = camera_params["lookat"]
    p3 = [p1[0], p1[1] + 1, p1[2]]
    a, b, c = get_abc_plane(p1, p2, p3)

    #want normal to be be on the "righthand" side of camera
    #x and z component of vec for the direction camera is pointing
    camera_pointing_x = camera_params["lookat"][0] - camera_params["camera"][0]
    camera_pointing_y = camera_params["lookat"][1] - camera_params["camera"][1]
    camera_pointing_z = camera_params["lookat"][2] - camera_params["camera"][2]

    result = np.cross(np.array([a, b, c]), np.array([camera_pointing_x, camera_pointing_y, camera_pointing_z]))
    #if y-component < 0, multiply by -1
    to_return = [-a, -b, -c] if result[2] < 0 else [a, b, c]

    return to_return


# Need camera parameters and vertical plane (so horizontal will be perpendicular to it)
def get_horizontal_plane(camera_params, a_vertical, b_vertical, c_vertical):
    p1 = camera_params["camera"]
    p2 = camera_params["lookat"]
    p3 = [p1[0] + a_vertical, p1[1] + b_vertical, p1[2] + c_vertical] #adding normal vector (a, b, c) to the point to make a third point on the horizontal plane
    a, b, c = get_abc_plane(p1, p2, p3)

    #make sure that this normal is upright, so b > 0
    if b < 0:
    	return -a, -b, -c
    elif b > 0:
    	return a, b, c

    print("uh oh. camera looking straight up or straight down")
    return a, b, c


# Given 3 points on a plane (p1, p2, p3), get a, b, and c coefficients in the general form.
# abc also gives a normal vector
def get_abc_plane(p1, p2, p3):
    # using Method 2 from wikipedia: https://en.wikipedia.org/wiki/Plane_(geometry)#:~:text=In%20mathematics%2C%20a%20plane%20is,)%20and%20three%2Ddimensional%20space.
	D = np.linalg.det(np.array([[p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]]]))
	if D == 0:
		print("crap! determinant D=0")
		print(p1)
		print(p2)
		print(p3)
		return

	# implicitly going to say d=-1 to obtain solution set
	a = np.linalg.det(np.array([[1, 1, 1], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]]])) / D
	b = np.linalg.det(np.array([[p1[0], p2[0], p3[0]], [1, 1, 1], [p1[2], p2[2], p3[2]]])) / D
	c = np.linalg.det(np.array([[p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [1, 1, 1]])) / D

	return a, b, c


def proj_vec_to_plane(a, b, c, x, y, z):
	num = a * x + b * y + z * c
	denom = math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2)
	constant = num / denom
	u_1 = x - constant * a
	u_2 = y - constant * b
	u_3 = z - constant * c
	return u_1, u_2, u_3


def label_to_int(label):
	if label.lower() in coco_dataset:
		return coco_dataset[label.lower()]

	return -1


def get_frame_truth(curr_frame, scene_truth):
	ground_truth_output = []

	params = {"image_dim_x": 320, "image_dim_y": 240, "horizontal_FoV": 60.0, "vertical_FoV": 40.0}

	for obj in scene_truth:
		obj_dict = {"semantic_label": None, "position": None}
		obj_dict["semantic_label"] = obj["semantic_label"]
		obj_dict["position"] = get_image_xy(curr_frame, params, obj["position"])

		# Only appends object that are in the coordinates of the frame.
		if obj_dict["position"][0] >= 0 and obj_dict["position"][0] <= params["image_dim_x"] and obj_dict["position"][1] >= 0 and obj_dict["position"][1] <= params["image_dim_y"]:
			ground_truth_output.append(obj_dict)

	return ground_truth_output


def format_ground_truth(ground_truth):
	ground_truth_output = []

	for item in ground_truth:
		item_dict = {"semantic_label": None, "position": None}
		item_dict["semantic_label"] = label_to_int(item["semantic_label"])
		item_dict["position"] = item["position"]

		ground_truth_output.append(item_dict)

	return ground_truth_output


# Handles frames
def frame(curr_frame, frame_count, scene_truth):
	frame_dict = {"camera": None, "lookat": None, "ground_truth": None, "image": curr_frame["image"]}
	detections_dict = {"labels": [], "center": [], "scores": [], "boxes": []}

	frame_dict["camera"] = [curr_frame["camera"][0], curr_frame["camera"][1], curr_frame["camera"][2]]
	frame_dict["lookat"] = [curr_frame["lookat"][0], curr_frame["lookat"][1], curr_frame["lookat"][2]]
	
	# Calculate the ground_truth for the frame
	frame_dict["ground_truth"] = get_frame_truth(curr_frame, scene_truth)

	return frame_dict


# Handles scenes
def scene(curr_scene, scene_count):
	scene_dict = {"views": [], "ground_truth": None}

	frame_count = 0

	scene_dict["ground_truth"] = format_ground_truth(curr_scene["labels"])

	# Iterate over all frames in the given scene
	for curr_frame in curr_scene["views"]:
		frame_dict = frame(curr_frame, frame_count, scene_dict["ground_truth"])
		scene_dict["views"].append(frame_dict)
		frame_count += 1

	return scene_dict


if __name__ == "__main__":
	# Load json
	data = json.load(open("/gpfs/loomis/project/jara-ettinger/mb2987/metagen_project_shared_space/Dataset/75views_per_scene/0_data.json", "rb"))

	# Output dict
	output = []

	scene_count = 0

	# Iterate over all scenes
	for curr_scene in data:

		scene_dict = scene(curr_scene, scene_count)

		output.append(scene_dict)

		scene_count += 1

	with open('75frames_with2d.json', 'w') as outfile:
		json.dump(output, outfile)
		exit()
