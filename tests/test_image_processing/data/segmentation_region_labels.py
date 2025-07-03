# Mapping of segmentation map region values to face region names
# Example: {region_value: "region_name"}
region_label_map = {
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "nose",
    7: "upper_lip",
    8: "lower_lip",
    9: "background",
    10: "forehead",
    # Add or adjust as needed based on your dataset
}

import json
with open("./data/segmentation_region_labels.json", "w") as f:
    json.dump(region_label_map, f, indent=4)
print("Saved region label mapping to ./data/segmentation_region_labels.json")
