import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue
import time
import math

# ----------------------------------------
# 1. Initialize MediaPipe Pose
# ----------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ----------------------------------------
# 2. Voice Assistant
# ----------------------------------------

speech_queue = queue.Queue(maxsize=2)


def voice_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)

    while True:
        message = speech_queue.get()

        try:
            if message is None:
                break

            engine.say(message)
            engine.runAndWait()

        except Exception as error:
            print("Voice error:", error)

        finally:
            speech_queue.task_done()

    engine.stop()


threading.Thread(
    target=voice_worker,
    daemon=True
).start()


def speak(message):
    # Avoid repeating old feedback when the queue is full.
    if speech_queue.full():
        return

    speech_queue.put_nowait(message)


# ----------------------------------------
# 3. Calculate Angle Between Three Points
# ----------------------------------------

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ----------------------------------------
# 4. Open Webcam
# ----------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    pose.close()
    raise SystemExit

rep_count = 0
stage = "up"

start_time = time.time()

last_feedback_time = 0
feedback_interval = 4

last_prompt = ""

# ----------------------------------------
# 5. Main Processing Loop
# ----------------------------------------

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        # Mirror the video for a natural experience.
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_frame)

        current_time = time.time()
        elapsed_time = int(current_time - start_time)

        feedback = "Stand straight to begin"

        # --------------------------------
        # 6. Detect Body Landmarks
        # --------------------------------

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Landmark visibility checks.
            required_points = [
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
                mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ]

            visible = all(
                landmarks[i].visibility > 0.6
                for i in required_points
            )

            if visible:

                # Extract left-side landmarks.
                hip = landmarks[
                    mp_pose.PoseLandmark.LEFT_HIP.value
                ]

                knee = landmarks[
                    mp_pose.PoseLandmark.LEFT_KNEE.value
                ]

                ankle = landmarks[
                    mp_pose.PoseLandmark.LEFT_ANKLE.value
                ]

                shoulder = landmarks[
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value
                ]

                hip_point = (hip.x, hip.y)
                knee_point = (knee.x, knee.y)
                ankle_point = (ankle.x, ankle.y)
                shoulder_point = (shoulder.x, shoulder.y)

                # --------------------------------
                # 7. Calculate Knee Angle
                # --------------------------------

                knee_angle = calculate_angle(
                    hip_point,
                    knee_point,
                    ankle_point
                )

                # --------------------------------
                # 8. Count Squat Repetitions
                # --------------------------------

                if knee_angle < 100:

                    stage = "down"
                    feedback = "Squat position detected"

                elif knee_angle > 160:

                    if stage == "down":
                        rep_count += 1
                        stage = "up"

                        speak("Good job! One repetition completed.")

                    else:
                        stage = "up"

                    feedback = "Stand straight"

                else:

                    feedback = "Keep moving smoothly"

                # --------------------------------
                # 9. Check Torso Lean
                # --------------------------------

                # This estimates forward lean from a
                # side-view camera, not from a front view.

                torso_angle = calculate_angle(
                    (shoulder.x, shoulder.y),
                    (hip.x, hip.y),
                    (hip.x, hip.y - 1)
                )

                if torso_angle > 35 and stage == "down":

                    feedback = "Keep your chest up"

                # --------------------------------
                # 10. Give Corrective Voice Feedback
                # --------------------------------

                if (
                    feedback != last_prompt
                    and current_time - last_feedback_time
                    >= feedback_interval
                    and feedback in [
                        "Keep your chest up",
                        "Keep moving smoothly"
                    ]
                ):

                    speak(feedback)

                    last_prompt = feedback
                    last_feedback_time = current_time

                # --------------------------------
                # 11. Draw Landmarks
                # --------------------------------

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Display angle and status.
                cv2.putText(
                    frame,
                    f"Knee Angle: {int(knee_angle)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Stage: {stage}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            else:
                feedback = "Move into full camera view"

        else:
            feedback = "No pose detected"

        # --------------------------------
        # 12. Display Reps, Timer and Feedback
        # --------------------------------

        cv2.putText(
            frame,
            f"Squat Reps: {rep_count}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        cv2.putText(
            frame,
            f"Time: {minutes:02d}:{seconds:02d}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            feedback,
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "AI Virtual Fitness Assistant",
            frame
        )

        # Press Q to exit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass

    print("Workout completed!")
    print("Total squat repetitions:", rep_count)