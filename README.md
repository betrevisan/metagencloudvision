# An Analysis of CloudVision for Object Detection

The content of this repository refers to the code used to conduct a comparitive analysis of Google's Cloud Vision API for object detection. The motivation behind it is to support the inference skills of MetaGen.

The add_2d_groundtruth.py file loads the scenenet dataset and adds a 2-dimensional ground truth to each frame based on the 3-dimensional ground truth for each scene.

The scenenet.py file loads the scenenet dataset with 2-dimensional ground truths and passes each frame to the CloudVision API for object detection.

Finally, the misrates_hallucinations.py file computes the metrics that evaluate the the artificial visual system's performance (misrate, hallucination rate, and misplacement rate).

All the work was done in collaboration with Marlene Berke, Tanushree Burman, and Zhangir Azerbayev.

May, 2021.
