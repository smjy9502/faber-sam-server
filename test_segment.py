import base64
import requests

IMAGE_PATH = "test.png"
POINT_X = 450
POINT_Y = 520

response = requests.post(
    "http://127.0.0.1:8000/segment",
    files={
        "image": open(IMAGE_PATH, "rb"),
    },
    data={
        "point_x": POINT_X,
        "point_y": POINT_Y,
    },
)

response.raise_for_status()
result = response.json()

mask_base64 = result["mask_png_base64"]
mask_bytes = base64.b64decode(mask_base64)

with open("sam_mask_result.png", "wb") as file:
    file.write(mask_bytes)

print("OK")
print("score:", result["score"])
print("bbox:", result["bbox"])
print("saved: sam_mask_result.png")