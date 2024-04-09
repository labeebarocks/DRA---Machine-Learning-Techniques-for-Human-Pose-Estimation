import cv2
import mediapipe as mp
from typing import List
import math
import numpy as np
import time
import threading
import RULA_tables 
import sys
from mediapipe.framework.formats import landmark_pb2

# Check if the user provided the path argument
if len(sys.argv) != 2:
    print("Usage: python script_name.py path_to_video")
    sys.exit(1)

video_path = sys.argv[1]

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Create a drawing object
mp_drawing = mp.solutions.drawing_utils

# Initialize scores
scores = {
    "upper_arm": 0,
    "lower_arm": 0,
    "wrist": 0,
    "wrist_twist": 0,
    "neck": 0,
    "trunk": 0,
    "leg": 0
}
# Initialize angles
angles = {
    "right_upper_arm": 0,
    "left_upper_arm": 0,
    "right_lower_arm": 0,
    "left_lower_arm": 0,
    "right_wrist": 0,
    "left_wrist": 0,
    "right_wrist_twist": 0,
    "left_wrist_twist": 0,
    "trunk": 0,
}
# Define a mapping for scores and corresponding landmarks
angle_mappings = {
    "left_lower_arm": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    "right_lower_arm": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),

    "left_upper_arm":(mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    "right_upper_arm":(mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),

    "left_wrist":(mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB),
    "right_wrist":(mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB),

    "trunk":(mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER)
}

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Middle point (e.g., elbow)
    c = np.array(c)  # Third point (e.g., wrist)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
# Function to update maximum scores
def update_angles(landmarks: List[landmark_pb2.NormalizedLandmark], angles):
    # Ensure landmarks list is not empty
    if not landmarks:
        return

    for score_name, landmark_indices in angle_mappings.items():
        # Extract specific landmarks for each angle calculation
        point1 = [landmarks[landmark_indices[0].value].x, landmarks[landmark_indices[0].value].y]
        point2 = [landmarks[landmark_indices[1].value].x, landmarks[landmark_indices[1].value].y]
        point3 = [landmarks[landmark_indices[2].value].x, landmarks[landmark_indices[2].value].y]

        # Calculate the angle
        angle = calculate_angle(point1, point2, point3)

        print(f"Angle for {score_name}: {angle}")  # Add this line

        # Update the score if the new angle is greater
        angles[score_name] = max(angles.get(score_name, 0), angle)

def update_scores(angles, scores):
   # Define scoring rules
    scoring_rules = {
        "upper_arm": [
            (lambda angle: angle < 20, 1),
            (lambda angle: 20 <= angle < 45, 2),
            (lambda angle: 45 <= angle < 90, 3),
            (lambda angle: angle >= 90, 4)
        ],
        "lower_arm": [
            (lambda angle: (angle <= 180 and angle >= 120) or angle >= 100, 2),
            (lambda angle: True, 1)  # Default case for all other angles
        ],
        "trunk": [
            (lambda angle: angle == 180, 1),
            (lambda angle: 160 <= angle < 180, 2),
            (lambda angle: 120 <= angle < 160, 3),
            (lambda angle: angle < 160, 4)
        ],
        "wrist": [
        (lambda angle: angle == 180, 1),
        (lambda angle: 165 <= angle < 180, 2),
        (lambda angle: angle < 165, 3),
        ]

    }

    # Apply scoring rules
    for part, rules in scoring_rules.items():
        part_angle = max(angles[f"right_{part}"], angles[f"left_{part}"]) if part != "trunk" else angles[part]

        print(f"Current angle for {part}: {part_angle}")  # Add this line

        for rule, score in rules:
            if rule(part_angle):
                scores[part] = max(1, scores[part] + score)  # Ensuring score does not go below 1
                break  # Stop checking further rules once one is met

def get_user_input(prompt, response_type=str, options=None):
    print(prompt)
    while True:
        response = input().lower()  # Convert to lower case for consistency
        if options and response not in options:
            print("Invalid input. Please enter one of the following: " + ", ".join(options))
        else:
            try:
                return response_type(response)
            except ValueError:
                print("Invalid input. Please try again.")

def update_user_input_scores(scores):
    # Define user input questions and their score adjustments
    user_input_questions = {
        "upper_arm": [
            ("Were your shoulders raised? (y/n)", lambda response: 1 if response == 'y' else 0, None),
            ("Was your upper arm abducted? (y/n)", lambda response: 1 if response == 'y' else 0, None),
            ("Were your arms supported or were you leaning on anything? (y/n)", lambda response: -1 if response == 'y' else 0, None),
        ],
        "lower_arm": [
            ("Were either of your arms working across your midline or out from your body? (y/n)", lambda response: 1 if response == 'y' else 0, None),
        ],
        "wrist": [
            ("Was your wrist bent from the midline? (y/n)", lambda response: 1 if response == 'y' else 0, None),
        ],
        "wrist_twist": [
            ("Were your palms facing up or down? (up/down)", lambda response: 2 if response == 'up' else 1 if response == 'down' else 0, ['up', 'down']),
        ],
        "neck": [
            ("Was your neck bent backwards? (y/n)", lambda response: 4 if response == 'y' else 0, None),
            ("Was your neck bent forwards? (Enter the angle in degrees)", lambda angle: 1 if 0 <= float(angle) < 10 else 2 if 10 <= float(angle) < 20 else 3 if float(angle) >= 20 else 0, None),
            ("Was your neck twisted? (y/n)", lambda response: 1 if response == 'y' else 0, None),
            ("Was your neck side bending? (y/n)", lambda response: 1 if response == 'y' else 0, None),
        ],
        "trunk": [
            ("Was your trunk twisted? (y/n)", lambda response: 1 if response == 'y' else 0, None),
            ("Was your trunk side bending? (y/n)", lambda response: 1 if response == 'y' else 0, None),
        ],
        "leg": [
            ("Were your legs and feet supported? (y/n)", lambda response: 1 if response == 'y' else 2, None),
        ],
    }


    # Iterate through questions and update scores
    for part, questions in user_input_questions.items():
        for question, score_func, options in questions:
            response = get_user_input(question, response_type=str, options=options)
            scores[part] += score_func(response)

def additional_calc(tablescore, body_part_type):
    ans = input(f"Are you maintaining that {body_part_type} posture or repeating the {body_part_type} movement more than 4 times? (Enter y/n) ")
    if ans == 'y':
        tablescore += 1
    ans = input(f"How much load/force is acting on your {body_part_type}? (Answer in lbs) ")
    if 4.4 <= float(ans) <= 22:
        follow_up = input("Is it a consistent or repeated force? (y/n) ")
        if follow_up == 'y':
            tablescore += 2
        else:
            tablescore += 1
    elif float(ans) > 22:
        tablescore += 3
    return tablescore

def calculate_RULA_score(scores):
    # Access scores from the dictionary
    upper_arm_score = scores["upper_arm"]
    lower_arm_score = scores["lower_arm"]
    wrist_score = scores["wrist"]
    wrist_twist_score = scores["wrist_twist"]
    neck_score = scores["neck"]
    trunk_score = scores["trunk"]
    leg_score = scores["leg"]

    # Assuming rula_table_a, rula_table_b, and rula_table_c are multidimensional arrays or nested dictionaries
    tableA_score = RULA_tables.rula_table_a[upper_arm_score][lower_arm_score][wrist_score][wrist_twist_score]
    tableB_score = RULA_tables.rula_table_b[neck_score][trunk_score][leg_score]

    wrist_arm_score = additional_calc(tableA_score, "arm")
    neck_trunk_leg_score = additional_calc(tableB_score, "neck and trunk")
    if wrist_arm_score >8 or neck_trunk_leg_score>7:
        wrist_arm_score = 8
        neck_trunk_leg_score = 7

    final_score = RULA_tables.rula_table_c[wrist_arm_score][neck_trunk_leg_score]

    return final_score


# Function to draw RULA score on the frame
def draw_RULA_score(frame, score):
    # Draw RULA score on the frame
    cv2.putText(frame, "RULA Score: {}".format(score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Function to process frame and perform RULA assessment
def process_frame(frame, angles, angle_mappings):
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose model
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for score_name, landmark_indices in angle_mappings.items():
            # Extract landmarks for each angle calculation
            point1 = results.pose_landmarks.landmark[landmark_indices[0].value]
            point2 = results.pose_landmarks.landmark[landmark_indices[1].value]
            point3 = results.pose_landmarks.landmark[landmark_indices[2].value]

            # Draw the angle calculation points
            for point in [point1, point2, point3]:
                cx, cy = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Display angle at the middle joint
            middle_x, middle_y = int(point2.x * frame.shape[1]), int(point2.y * frame.shape[0])
            
            # Recalculate the angle
            recalculated_angle = calculate_angle([point1.x, point1.y], [point2.x, point2.y], [point3.x, point3.y])
            angles[score_name] = recalculated_angle

            # Round the angle value for display
            angle_display = round(recalculated_angle)

            # Draw the angle value near the middle joint
            cv2.putText(frame, str(angle_display), (middle_x, middle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame


# Main function to capture video feed and perform RULA assessment
def main():
    # Initialize video capture with the provided video file
    cap = cv2.VideoCapture(video_path)

    # Get frame width, height, and FPS from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, scores, angle_mappings)

        # Write the processed frame into the output file
        out.write(processed_frame)

        cv2.imshow('RULA Assessment', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()

    # Update the scores based on the calculated angles
    update_scores(angles, scores)

    # Gather additional user input and adjust scores accordingly
    update_user_input_scores(scores)

    print("Updated Scores:", scores)
    # Calculate the final RULA score
    final_rula_score = calculate_RULA_score(scores)

    # Return or print the final RULA score
    print("Final RULA Score:", final_rula_score)
    return final_rula_score

# Call the main function to start the program
if __name__ == "__main__":
    final_score = main()
    print("The RULA assessment's final score is:", final_score)

