import json


# Returns True if the object was detected, otherwise returns False.
def detected(obj, frame_detections):
	for d in frame_detections["labels"]:
		if obj["semantic_label"] == d:
			return True

	return False


# Returns True if the object was hallucinated, otherwise returns False.
def hallucinated(obj, frame_ground_truth):
	for t in frame_ground_truth:
		if t["semantic_label"] == obj:
			return False

	return True


# Returns True if the object was detected at the wrong location, otherwise returns False.
def wrongly_located(label, location, frame_ground_truth):
	for t in frame_ground_truth:
		if t["semantic_label"] == label:
			if location[0] - 80 > t["position"][0] or location[0] + 80 < t["position"][0] or location[1] - 60 > t["position"][1] or location[1] + 60 < t["position"][1]:
				return True

	return False


# Computes the total misrate for all scenes.
def compute_misrate(detections):
	total_scene_misrate = 0

	# Iterate over every scene
	for scene in detections:

		# Iterate over every frame
		total_frame_misrate = 0
		for frame in scene["views"]:

			# Count how many of the objects in the ground truth were detected
			detected_count = 0
			for obj in frame["ground_truth"]:
				if detected(obj, frame["detections"]):
					detected_count += 1

			misrate = 1
			if len(frame["ground_truth"]) != 0:
				misrate = detected_count / len(frame["ground_truth"])
			
			total_frame_misrate += misrate

		total_scene_misrate += total_frame_misrate / len(scene["views"])

	return total_scene_misrate / len(detections)


# Computes the total hallucination rate for all scenes.
def compute_hallucination(detections):
	total_scene_hallucination = 0

	# Iterate over every scene
	for scene in detections:

		# Iterate over every frame
		total_frame_hallucination = 0
		for frame in scene["views"]:

			# Count how many objects were detected that were not in the frame
			hallucinated_count = 0
			for obj in frame["detections"]["labels"]:
				if hallucinated(obj, frame["ground_truth"]):
					hallucinated_count += 1

			hallucination = 0
			if len(frame["detections"]["labels"]) != 0:
				hallucination = hallucinated_count / len(frame["detections"]["labels"])
			
			total_frame_hallucination += hallucination

		total_scene_hallucination += total_frame_hallucination / len(scene["views"])

	return total_scene_hallucination / len(detections)


# Computes the total misplacement rate for all scenes.
def compute_wrong_location(detections):
	total_scene_wrong_location = 0

	# Iterate over every scene
	for scene in detections:

		# Iterate over every frame
		total_frame_wrong_location = 0
		for frame in scene["views"]:

			# Count how many objects were detected that were not in the frame
			wrong_location_count = 0
			for label, location in zip(frame["detections"]["labels"], frame["detections"]["center"]):
				if wrongly_located(label, location, frame["ground_truth"]):
					wrong_location_count += 1

			wrong_location = 0
			if len(frame["detections"]["labels"]) != 0:
				wrong_location = wrong_location_count / len(frame["detections"]["labels"])
			
			total_frame_wrong_location += wrong_location

		total_scene_wrong_location += total_frame_wrong_location / len(scene["views"])

	return total_scene_wrong_location / len(detections)


if __name__ == "__main__":
	# Load json
	detections = json.load(open("/gpfs/loomis/project/jara-ettinger/mb2987/metagen_project_shared_space/CloudVision/detections.json", "rb"))

	# Output dict
	output = {"misrate": 0, "hallucination": 0, "wrong_location": 0, "scene_count": 0}

	output["scene_count"] = len(detections);
	print(output["scene_count"])

	output["misrate"] = compute_misrate(detections);
	print(output["misrate"])

	output["hallucination"] = compute_hallucination(detections);
	print(output["hallucination"])

	output["wrong_location"] = compute_wrong_location(detections);
	print(output["wrong_location"])
