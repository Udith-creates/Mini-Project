import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

# Initialize model once
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX


def startapplication(video_source=0):
    """
    Start the application using the given video_source.

        video_source can be:
            - an integer (0, 1, ...) for camera devices
            - a string path to a video file (e.g. 'myclip.mp4')
    """
    # Accept numeric strings (from CLI) as ints
    try:
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
    except Exception:
        pass

    video = cv2.VideoCapture(video_source)
    if not video.isOpened():
        print(f"Error: cannot open video source {video_source}")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            # End of file or camera error — stop the loop
            break

        # Convert and resize for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob_val = round(prob[0][0] * 100, 2)

            # to beep when alert (platform dependent)
            # if(prob_val > 90):
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob_val}", (20, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # default 0 (webcam) — change to a filename to use a file
    startapplication(0)
