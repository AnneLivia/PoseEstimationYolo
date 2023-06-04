# pose estmation is a task that returns the location of specific points called keypoints.
# Keypoints can represent various parts such as joints, landmarks and other features.
# the locations are usually represented as a set of 2D [x, y] or 3D [x, y, visible] coordinates.
# documentation: https://docs.ultralytics.com/modes/predict/#plotting-results
from ultralytics import YOLO;
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vi', required=True, help='Provide the relative path for the video')
args = parser.parse_args()

# change resolution of the video
outputPath = './video/resized_running.mp4'

def resize_video():
    # reading video
    video = cv2.VideoCapture(args.vi);
    fps = video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(outputPath, fourcc, fps=fps, frameSize=(680, 420))
    while video.isOpened():
        sucess, frame = video.read()
        if not sucess:
            break;
        
        resized_frame = cv2.resize(frame, (680, 420))
        videoWriter.write(resized_frame)

    video.release()

def pose_estimation():
    # need to import the pre-trained model to perform the pose estimation
    poseModel = YOLO('./model/YOLOv8n-pose.pt')

    ## code bellow peforms pose estimation directly to the video and save the result
    poseModel.predict(outputPath, conf=0.25, device='cpu', save=True, show=True, line_width=0)

# calling the function to resize video
resize_video()
pose_estimation()

cv2.destroyAllWindows()