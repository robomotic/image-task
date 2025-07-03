from image_segmentation.app.version import __version__
import base64
import json
import os

def test_get_version(test_client):
    response = test_client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json() == {"version": __version__}


def test_frontal_crop_submit(test_client):
    """
    These files are loaded
    """
    # Check that all required files exist
    assert os.path.exists("./data/landmarks.txt"), "landmarks.txt is missing"
    assert os.path.exists("./data/original_image.png"), "original_image.png is missing"
    assert os.path.exists("./data/segmentation_map.png"), "segmentation_map.png is missing"

    with open("./data/landmarks.txt", "r") as file:
        landmarks_str = file.read()
        try:
            landmarks = json.loads(landmarks_str)
        except json.JSONDecodeError:
            # Try to convert single quotes to double quotes for JSON compatibility
            import ast
            landmarks = ast.literal_eval(landmarks_str)

    with open("./data/original_image.png", "rb") as file:
        testimage = file.read()
    image_b64 = base64.b64encode(testimage).decode("utf-8")

    with open("./data/segmentation_map.png", "rb") as file:
        segmap = file.read()
    segmap_b64 = base64.b64encode(segmap).decode("utf-8")

    payload = {
        "image": image_b64,  # base64 for 'testimage'
        "landmarks": landmarks["landmarks"][0],
        "segmentation_map": segmap_b64,  # base64 for 'segmap'
    }
    response = test_client.post("/api/v1/frontal/crop/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "svg" in data
    assert "mask_contours" in data
    assert isinstance(data["svg"], str)
    assert isinstance(data["mask_contours"], dict)
