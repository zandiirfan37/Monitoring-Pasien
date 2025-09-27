
import cv2

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

if __name__ == '__main__':
    cameras = list_cameras()
    if cameras:
        print(f"Available camera indices: {cameras}")
    else:
        print("No cameras found.")
