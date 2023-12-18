import cv2
import mediapipe as mp
import os
# from cvzone.HandTrackingModule import HandDetector
import pyttsx3
from handTracker import HandDetector
import numpy as np

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
    print("Commencing Presentation...")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say("Just a moment, We are starting the virtual presentation now.")
    engine.runAndWait()

    # variable
    imageNumber = 0
    widthSlide, heightSlide = 1280, 720
    widthCam, heightCam = int(120 * 2), int(100 * 2)
    gestureThreshold = 300
    buttonPressed = False
    buttonCounter = 0
    buttonDelay = 30
    annotations = [[]]
    annotationNumber = -1
    annotationStart = False

    # define the path of the presentation folder
    folderPath = 'Presentation'

    # cam setup
    cap = cv2.VideoCapture(0)
    cap.set(3, widthSlide)
    cap.set(4, heightSlide)

    # GET list of presentation images
    pathImages = sorted(os.listdir(folderPath), key=len)
    # print(pathImages)

    # Hand Detector
    detector = HandDetector(detectionCon=0.9, maxHands=1)

    while True:
        # importing Images
        success, img = cap.read()
        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[imageNumber])
        imgCurrent = cv2.imread(pathFullImage)

        hands, img = detector.findHands(img, flipType=False)

        # Gesture Position line
        cv2.line(img, (0, gestureThreshold), (widthSlide, gestureThreshold), (0, 255, 0, 0.5), 5)

        if hands and buttonPressed == False:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            # print(fingers)

            # Accept gesture
            cx, cy = hand['center']  # if the hand present inside gesture box
            lmList = hand['lmList']

            # Constrain values for easier drawing
            xVal = int(np.interp(lmList[8][0], [widthSlide // 2, widthSlide], [0, widthSlide]))
            yVal = int(np.interp(lmList[8][1], [150, heightSlide - 150], [0, heightSlide]))
            indexFinger = xVal, yVal

            if cy <= gestureThreshold:

                # Gesture 1
                if fingers == [0, 0, 0, 0, 0]:  # thumb gesture/ previous slide
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    """ 
                        print('left')
                        break
                    """
                    if imageNumber > 0:
                        buttonPressed = True
                        imageNumber -= 1

                # Gesture 2
                if fingers == [1, 0, 0, 0, 1]:  # little finger gesture/ next slide
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    """
                        print('right')
                        break
                    """
                    if imageNumber < len(pathImages) - 1:
                        buttonPressed = True
                        imageNumber += 1

            # Gesture 3
            if fingers == [1, 1, 1, 0, 0]:  # index and middle finger gesture/ pointer
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            # Gesture 4
            if fingers == [1, 1, 0, 0, 0]:  # only index finger gesture/ draw
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                annotations[annotationNumber].append(indexFinger)
            else:
                annotationStart = False

            # Gesture 5
            if fingers == [1, 1, 1, 1, 0]:  # three finger gesture/ erase
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
        else:
            annotationStart = False


        # buttonPressed iterations
        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay:
                buttonCounter = 0
                buttonPressed = False

        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                if j != 0:
                    cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

                    # adding cam img in slide
        imgCam = cv2.resize(img, (widthCam, heightCam))
        h, w, _ = imgCurrent.shape
        imgCurrent[0:heightCam, w - widthCam:w] = imgCam

        # cv2.imshow("Image",img)
        cv2.imshow("Slides", imgCurrent)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

def open_calculator():
    # class calculator:
    #     def __init__(self, pos, width, height, value):
    #         self.pos = pos
    #         self.width = width
    #         self.height = height
    #         self.value = value
    #
    #     def drawbutton(self, img):
    #         cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
    #                       (125, 125, 225), cv2.FILLED)
    #         cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
    #                       (50, 50, 50), 3)
    #         cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN,
    #                     2, (50, 50, 50), 2)
    #
    #     def click(self, x, y):
    #         if self.pos[0] < x < self.pos[0] + self.width and \
    #                 self.pos[1] < y < self.pos[1] + self.height:
    #             cv2.rectangle(img, (self.pos[0] + 3, self.pos[1] + 3),
    #                           (self.pos[0] + self.width - 3, self.pos[1] + self.height - 3),
    #                           (255, 255, 255), cv2.FILLED)
    #             cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN,
    #                         5, (0, 0, 0), 5)
    #             return True
    #         else:
    #             return False
    #
    # engine = pyttsx3.init()
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[0].id)
    # engine.say("please wait! the virtual calculator is starting")
    # engine.runAndWait()
    #
    # buttons = [['7', '8', '9', 'C'],
    #            ['4', '5', '6', '*'],
    #            ['1', '2', '3', '+'],
    #            ['0', '-', '/', '='],
    #            ['(', ')', '.', 'del']]
    # buttonList = []
    # for x in range(4):
    #     for y in range(5):
    #         xpos = x * 100 + 700
    #         ypos = y * 100 + 100
    #         buttonList.append(calculator((xpos, ypos), 100, 100, buttons[y][x]))
    #
    # Equation = ''
    # Counter = 0
    # # Webcam
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(3, 1280)
    # cap.set(4, 1080)
    # detector = HandDetector(detectionCon=0.9, maxHands=1)
    #
    # while True:
    #     success, img = cap.read()
    #     img = cv2.flip(img, 1)
    #     hands, img = detector.findHands(img)
    #     for button in buttonList:
    #         button.drawbutton(img)
    #
    #     # Check for Hand
    #     if hands:
    #         # Find distance between fingers
    #         lmList = hands[0]['lmList']
    #         length, _, img = detector.findDistance(lmList[8], lmList[12], img)
    #         # print(length)
    #         x, y = lmList[8]
    #
    #         # If clicked check which button and perform action
    #         if length < 50 and Counter == 0:
    #             for i, button in enumerate(buttonList):
    #                 if button.Click(x, y):
    #                     myValue = buttons[int(i % 5)][int(i / 5)]  # get correct number
    #                     # engine.say(myValue)
    #                     # engine.runAndWait()
    #                     if myValue == '=':
    #                         try:
    #                             Equation = str(eval(Equation))
    #                         except SyntaxError:
    #                             print("Syntax Error")
    #                             engine.say("Syntax Error")
    #                             engine.runAndWait()
    #                             Equation = 'Syntax Error'
    #                     elif Equation == 'Syntax Error':
    #                         Equation = ''
    #                     elif myValue == 'C':
    #                         Equation = ''
    #                     elif myValue == 'del':
    #                         Equation = Equation[:-1]
    #                     else:
    #                         Equation += myValue
    #                     Counter = 1
    #
    #     # to avoid multiple clicks
    #     if Counter != 0:
    #         Counter += 1
    #         if Counter > 10:
    #             Counter = 0
    #
    #     # Final answer
    #     cv2.rectangle(img, (700, 20), (1100, 100),
    #                   (175, 125, 155), cv2.FILLED)
    #
    #     cv2.rectangle(img, (700, 20), (1100, 100),
    #                   (50, 50, 50), 3)
    #     cv2.putText(img, Equation, (710, 80), cv2.FONT_HERSHEY_PLAIN,
    #                 3, (0, 0, 0), 3)
    #     cv2.putText(img, 'VIRTUAL CALCULATOR -->', (50, 50), cv2.FONT_HERSHEY_PLAIN,
    #                 3, (0, 0, 0), 3)
    #     cv2.imshow("Virtual Calculator", img)
    #     cv2.moveWindow("Virtual Calculator", 0, 0)
    #     # close the webcam
    #     if cv2.waitKey(10) & 0xFF == ord("q"):
    #         break

    from calculator import Calculator

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say("Please wait! The virtual calculator is starting")
    engine.runAndWait()

    buttons = [['7', '8', '9', 'C'],
               ['4', '5', '6', '*'],
               ['1', '2', '3', '+'],
               ['0', '-', '/', '='],
               ['(', ')', '.', 'del']]
    buttonList = []
    for x in range(4):
        for y in range(5):
            xpos = x * 100 + 700
            ypos = y * 100 + 100
            buttonList.append(Calculator((xpos, ypos), 100, 100, buttons[y][x]))

    Equation = ''
    Counter = 0
    # Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 1080)
    detector = HandDetector(detectionCon=0.9, maxHands=1)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)
        for button in buttonList:
            button.drawbutton(img)

        # Check for Hand
        if hands:
            # Find distance between fingers
            lmList = hands[0]['lmList']
            length, _, img = detector.findDistance(lmList[8], lmList[12], img)
            # print(length)
            x, y = lmList[8]

            # If clicked check which button and perform action
            if length < 50 and Counter == 0:
                for i, button in enumerate(buttonList):
                    if button.Click(x, y):
                        myValue = buttons[int(i % 5)][int(i / 5)]  # get correct number
                        # engine.say(myValue)
                        # engine.runAndWait()
                        if myValue == '=':
                            try:
                                Equation = str(eval(Equation))
                            except SyntaxError:
                                print("Syntax Error")
                                engine.say("Syntax Error")
                                engine.runAndWait()
                                Equation = 'Syntax Error'
                        elif Equation == 'Syntax Error':
                            Equation = ''
                        elif myValue == 'C':
                            Equation = ''
                        elif myValue == 'del':
                            Equation = Equation[:-1]
                        else:
                            Equation += myValue
                        Counter = 1

        # to avoid multiple clicks
        if Counter != 0:
            Counter += 1
            if Counter > 10:
                Counter = 0

        # Final answer
        cv2.rectangle(img, (700, 20), (1100, 100),
                      (175, 125, 155), cv2.FILLED)

        cv2.rectangle(img, (700, 20), (1100, 100),
                      (50, 50, 50), 3)
        cv2.putText(img, Equation, (710, 80), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 0, 0), 3)
        cv2.putText(img, 'VIRTUAL CALCULATOR :', (50, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 0, 0), 3)
        cv2.imshow("Virtual Calculator", img)
        cv2.moveWindow("Virtual Calculator", 0, 0)
        # close the webcam
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Cleanup and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


def show_about():
    print("Displaying Project Information...")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say("Please hold on. Presenting information about our project.")
    engine.runAndWait()

    # Load the image
    image = cv2.imread(r"C:\Users\shrad\PycharmProjects\pythonProject\Gesture-Controlled-Virtual-Mouse-main\car.jpg")

    # Create a blank image to accommodate the image and caption
    height, width = image.shape[:2]
    output = np.zeros((height + 50, width, 3), dtype=np.uint8)

    # Copy the original image onto the output image
    output[:height, :] = image

    # Add the caption text to the output image
    caption = "This is the caption"
    cv2.putText(output, caption, (10, height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the output image
    cv2.imshow('Image with Caption', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Or, save the output image to a file
    cv2.imwrite('output.jpg', output)


def open_camera():
    print("Opening camera...")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say("Please wait while we showcase our project, Our virtual presentation is about to begin.")
    engine.runAndWait()

    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Failed to open the camera.")
        return
    # Create a face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a hand tracking object
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9)

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
        add_text_with_border(resized_frame, "Calculator", (50, 130), 1.5, (255, 192, 203), 2)
        add_text_with_border(resized_frame, "About", (50, 210), 1.5, (255, 192, 203), 2)

        # Draw faces on the frame
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check for hand landmarks and gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_x = int(index_finger_tip.x * FRAME_WIDTH)
                index_finger_tip_y = int(index_finger_tip.y * FRAME_HEIGHT)

                # Check if the index finger is on any of the text
                if 50 <= index_finger_tip_x <= 250 and 30 <= index_finger_tip_y <= 80:
                    start_presentation()
                elif 50 <= index_finger_tip_x <= 230 and 80 <= index_finger_tip_y <= 130:
                    open_calculator()
                elif 50 <= index_finger_tip_x <= 150 and 130 <= index_finger_tip_y <= 180:
                    show_about()

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(resized_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the resulting frame
        cv2.imshow('Virtual Presentation', resized_frame)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


# Call the function to open the camera
open_camera()
