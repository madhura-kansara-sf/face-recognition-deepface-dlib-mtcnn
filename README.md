Face Recognition Project
This project utilizes face recognition techniques to identify known individuals in live camera feeds. It employs the MTCNN (Multi-Task Cascaded Convolutional Neural Network) for face detection, the Dlib model for face encoding and DeepFace library for face recognition.

Dependencies
Python 3.x
OpenCV (cv2)
NumPy
Face Recognition (face_recognition)
DeepFace
MTCNN
PIL (Python Imaging Library)

Usage
Install the required dependencies using pip install -r requirements.txt.
Prepare a directory containing images of known individuals (known_people_dir) and ensure each person has their own subdirectory.
Run main.py to start the face recognition system. Make sure your webcam is connected and functional.
The system will continuously process frames from the webcam, detecting faces and attempting to match them with known individuals.
Detected faces will be outlined with a box, and their names (if recognized) will be displayed below.

Customization
You can adjust the threshold for face matching by changing the value in the code (0.07 by default).
Additional data augmentation techniques can be applied by modifying the augment_data() function in utils.py.

Acknowledgments
This project utilizes the DeepFace library for face recognition.
Face detection is performed using the MTCNN model.
