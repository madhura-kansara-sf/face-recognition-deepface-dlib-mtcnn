# Face Recognition Project

This project implements face recognition techniques to identify known individuals in live camera feeds. It utilizes the MTCNN (Multi-Task Cascaded Convolutional Neural Network) for face detection, the Dlib model for face encoding and DeepFace library for face recognition.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- Face Recognition (`face_recognition`)
- DeepFace
- MTCNN
- PIL (Python Imaging Library)

Install the required dependencies using `pip install -r requirements.txt`.

## Usage

1. Run `main.py` to start the face recognition system.
2. Ensure your webcam is connected and functional.
3. The system will continuously process frames from the webcam, detecting faces and attempting to match them with known individuals.
4. Detected faces will be outlined with a box, and their names (if recognized) will be displayed below.
5. Press 'q' to quit the application.
   
## Customization

- Adjust the threshold for face matching by changing the value in the code (`0.07` by default).
- Additional data augmentation techniques can be applied by modifying the `augment_data()` function in `utils.py`.

## Structure

- `main.py`: Main script for face detection and recognition.
- `utils.py`: Utility functions for image processing and data augmentation.

## Acknowledgments

- This project utilizes the DeepFace library for face recognition.
- Face detection is performed using the MTCNN model.
