from ultralytics.data.converter import convert_coco

convert_coco("./data/CrowdPose", use_keypoints=True)
