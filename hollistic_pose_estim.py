import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic model.
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize MediaPipe drawing utils for drawing the pose annotations.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_holistic(image, holistic):
    # Convert the color space from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect holistic data (pose, hands, face).
    results = holistic.process(image)

    # Convert back to BGR for OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_holistic(image, results):
    # Draw the pose annotations on the image.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Drawing face landmarks on the image
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())


def get_pose_caption(results):
    if results.pose_landmarks:
        # Extract landmarks for wrists, shoulders, and hips.
        left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
        left_pinky = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY]
        right_pinky = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY]
        left_index = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX]
        right_index = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX]
        left_thumb = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB]
        right_thumb = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB]
        left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
        right_mouth_corner = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT]
        left_mouth_corner = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT]


        # Thresholds for hand positions.
        vertical_threshold = 0.1
        horizontal_threshold = 0.1

        # Check if hands are raised.
        is_left_hand_raised = left_wrist.y < left_shoulder.y
        is_right_hand_raised = right_wrist.y < right_shoulder.y

        # Check if hands are on hips.
        is_left_hand_on_hip = abs(left_wrist.x - left_hip.x) < horizontal_threshold and \
                              abs(left_wrist.y - left_hip.y) < vertical_threshold
        is_right_hand_on_hip = abs(right_wrist.x - right_hip.x) < horizontal_threshold and \
                               abs(right_wrist.y - right_hip.y) < vertical_threshold
        
        # Check if hands are on mouth.
        is_right_hand_on_mouth = (abs(right_mouth_corner.x - right_wrist.x) < horizontal_threshold and \
                              abs(right_mouth_corner.y - right_wrist.y) < vertical_threshold) or \
                            (abs(right_mouth_corner.x - right_index.x) < horizontal_threshold and \
                              abs(right_mouth_corner.y - right_index.y) < vertical_threshold) or \
                            (abs(right_mouth_corner.x - right_thumb.x) < horizontal_threshold and \
                              abs(right_mouth_corner.y - right_thumb.y) < vertical_threshold) or \
                            (abs(right_mouth_corner.x - right_pinky.x) < horizontal_threshold and \
                              abs(right_mouth_corner.y - right_pinky.y) < vertical_threshold)
        
        is_left_hand_on_mouth = (abs(left_mouth_corner.x - left_wrist.x) < horizontal_threshold and \
                               abs(left_mouth_corner.y - left_wrist.y) < vertical_threshold) or \
                            (abs(left_mouth_corner.x - left_index.x) < horizontal_threshold and \
                              abs(left_mouth_corner.y - left_index.y) < vertical_threshold) or \
                            (abs(left_mouth_corner.x - left_thumb.x) < horizontal_threshold and \
                              abs(left_mouth_corner.y - left_thumb.y) < vertical_threshold) or \
                            (abs(left_mouth_corner.x - left_pinky.x) < horizontal_threshold and \
                              abs(left_mouth_corner.y - left_pinky.y) < vertical_threshold)
        
        # Check if arms are crossed

        are_arms_crossed = (abs(right_elbow.x - left_index.x) < horizontal_threshold and \
                               abs(right_elbow.y - left_index.y) < vertical_threshold) and \
                            (abs(left_elbow.x - right_index.x) < horizontal_threshold and \
                              abs(left_elbow.y - right_index.y) < vertical_threshold)

        # Determine the pose.
        if is_right_hand_on_mouth and is_left_hand_on_mouth:
            return "Both Hands on Mouth"
        elif is_left_hand_on_mouth:
            return "Right Hand on Mouth"
        elif is_right_hand_on_mouth:
            return "Left Hand on Mouth"
        elif is_left_hand_raised and is_right_hand_raised:
            return "Both Hands Raised"
        elif is_left_hand_raised:
            return "Right Hand Raised"
        elif is_right_hand_raised:
            return "Left Hand Raised"
        elif is_left_hand_on_hip and is_right_hand_on_hip:
            return "Hands on Hips"
        elif are_arms_crossed:
            return "Arms Crossed"
        #elif is_left_hand_on_hip:
            return "Left Hand on Hip"
        #elif is_right_hand_on_hip:
            return "Right Hand on Hip"
        else:
            return "Neutral Pose"
    return ""

def main():
    # Open the default camera for capturing video.
    cap = cv2.VideoCapture(0)
    print(cap)

    # Retrieve the width and height of the video frame.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the video codec and create a VideoWriter object to save the output.
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    out = cv2.VideoWriter('output.avi', fourcc, 20, (frame_width, frame_height))

    # Create a named window for displaying the output.
    cv2.namedWindow("Pose Detection")

    # Main loop for capturing and processing each frame.
    while True:
        # Read a frame from the camera.
        ret, frame = cap.read()
        
        # If frame reading was not successful, break from the loop.
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect.
        frame = cv2.flip(frame, 1)
        
        # Detect holistic data (pose, face, hands) in the frame.
        frame, results = detect_holistic(frame, holistic)

        # Draw the holistic annotations on the frame.
        draw_holistic(frame, results)

        # Get a caption based on the pose detected in the frame.
        caption = get_pose_caption(results)
        
        # If a caption is returned, overlay it on the frame.
        if caption:
            cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the processed frame into the output file.
        out.write(frame)
        
        # Display the frame in the named window.
        cv2.imshow("Pose Detection", frame)

        # Check if the window is closed or if the 'Esc' key is pressed.
        if cv2.getWindowProperty("Pose Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Break the loop if 'Esc' is pressed.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the camera and output file and close all OpenCV windows.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

