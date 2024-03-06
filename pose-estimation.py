import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utils for drawing the pose annotations.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_pose(image, pose):
    # Convert the color space from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose.
    results = pose.process(image)

    # Convert back to BGR for OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_pose(image, results):
    # Draw the pose annotations on the image.
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def get_pose_caption(results):
    if results.pose_landmarks:
        # Extract landmarks for wrists, shoulders, and hips.
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        face_ave = abs(mouth_left.x +mouth_right.x)/2

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
        
        is_right_hand_on_face = abs(right_wrist.x - face_ave) < horizontal_threshold

        is_left_hand_on_face = abs(left_wrist.x - face_ave) < horizontal_threshold

        # Determine the pose.
        if is_left_hand_raised and is_right_hand_raised:
            return "Both Hands Raised"
        elif is_left_hand_raised:
            return "Right Hand Raised"
        elif is_right_hand_raised:
            return "Left Hand Raised"
        elif is_right_hand_on_face:
            return "Right Hand on Face"
        elif is_left_hand_on_face:
            return "Left Hand on Face"
        elif is_left_hand_on_hip and is_right_hand_on_hip:
            return "Hands on Hips"
        #elif is_left_hand_on_hip:
            return "Left Hand on Hip"
        #elif is_right_hand_on_hip:
            return "Right Hand on Hip"
        else:
            return "Neutral Pose"
    return ""



def main():
    # Open the default camera.
    cap = cv2.VideoCapture(0)

    # Get the width and height of the frame from the camera for VideoWriter.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object.
    # The output is stored in 'output.avi' file.
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    out = cv2.VideoWriter('output.avi', fourcc, 20, (frame_width, frame_height))


    # Create a window.
    cv2.namedWindow("Pose Detection")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally.
        frame = cv2.flip(frame, 1)
        
        # Detect the pose.
        frame, results = detect_pose(frame, pose)

        # Draw the pose.
        draw_pose(frame, results)

        # Get the pose caption.
        caption = get_pose_caption(results)
        if caption:
            cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame into the file 'output.avi'.
        out.write(frame)
        
        # Show the frame.
        cv2.imshow("Pose Detection", frame)

        # Break the loop if the window is closed.
        if cv2.getWindowProperty("Pose Detection", cv2.WND_PROP_VISIBLE) < 1:  
            break

        # Break the loop if 'Esc' is pressed.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
