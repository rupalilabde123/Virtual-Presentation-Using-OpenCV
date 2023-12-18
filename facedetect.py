import cv2
import mediapipe as mp

FRAME_WIDTH = 1366
FRAME_HEIGHT = 768

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def resize_frame(frame):
    # Resize the frame to the desired dimensions
    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    return resized_frame

def add_text_with_border(image, text, position, font_scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the coordinates for the text and the border
    text_x, text_y = position
    border_x = text_x - thickness
    border_y = text_y - text_size[1] - thickness

    # Add the black border
    cv2.rectangle(image, (border_x, border_y), (border_x + text_size[0] + thickness, border_y + text_size[1] + thickness), (0, 0, 0), -1)

    # Add the pink text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

def start_presentation():
    print("Starting presentation...")

def calculator():
    print("Opening calculator...")

def open_camera():
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Failed to open the camera.")
        return

    # Create a face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a hand tracking object
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret will be True
        if not ret:
            print("Failed to receive frame from the camera.")
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Resize the frame to desired dimensions
        resized_frame = resize_frame(flipped_frame)

        # Convert the frame to RGB for hand tracking
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process the frame with hand tracking
        results = hands.process(frame_rgb)

        # Add the text with border
        add_text_with_border(resized_frame, "Start Presentation", (50, 50), 1.5, (255, 192, 203), 2)
        add_text_with_border(resized_frame, "Calculator", (50, 150), 1.5, (255, 192, 203), 2)
        add_text_with_border(resized_frame, "About", (50, 250), 1.5, (255, 192, 203), 2)

        # Draw faces on the frame
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(resized_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the resulting frame
        cv2.imshow('Hand Tracking', resized_frame)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            start_presentation()
        elif key == ord('c'):
            calculator()

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

open_camera()
