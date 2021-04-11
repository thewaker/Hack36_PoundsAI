import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0)

file_list = ["IMG-0069.jpg", "IMG-0070.jpg", "IMG-0071.jpg", "IMG-0072.jpg"]

# For static images:

for idx, file in enumerate(file_list):
  image = cv2.imread(file)
  image_height, image_width, _ = image.shape
  # Convert the BGR image to RGB before processing.
  results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  if not results.pose_landmarks:
    continue

  #print(results.pose_landmarks)
  #print("done")

  #  11-12 (left_shoulder, right_shoulder)   
  x_11_12 = results.pose_landmarks.landmark[11].x - results.pose_landmarks.landmark[12].x
  y_11_12 = results.pose_landmarks.landmark[11].y - results.pose_landmarks.landmark[12].y
  z_11_12 = results.pose_landmarks.landmark[11].z - results.pose_landmarks.landmark[12].z

  #  11-23 (left_shoulder, left_hip)
  x_11_23 = results.pose_landmarks.landmark[11].x - results.pose_landmarks.landmark[23].x
  y_11_23 = results.pose_landmarks.landmark[11].y - results.pose_landmarks.landmark[23].y
  z_11_23 = results.pose_landmarks.landmark[11].z - results.pose_landmarks.landmark[23].z
  
  #  12-24
  x_12_24 = results.pose_landmarks.landmark[12].x - results.pose_landmarks.landmark[24].x
  y_12_24 = results.pose_landmarks.landmark[12].y - results.pose_landmarks.landmark[24].y
  z_12_24 = results.pose_landmarks.landmark[12].z - results.pose_landmarks.landmark[24].z
  #  24-23 (left_hip, left_hip)
  x_23_24 = results.pose_landmarks.landmark[23].x - results.pose_landmarks.landmark[24].x
  y_23_24 = results.pose_landmarks.landmark[23].y - results.pose_landmarks.landmark[24].y
  z_23_24 = results.pose_landmarks.landmark[23].z - results.pose_landmarks.landmark[24].z

  # angle between shoulder and hip 
  a_sh = math.acos((x_11_12 * x_23_24 + y_11_12 * y_23_24 + z_11_12 * z_23_24)/(((x_11_12 ** 2)+(y_11_12 ** 2)+(z_11_12 ** 2))**(1/2) * ((x_23_24 **2)+(y_23_24 ** 2)+(z_23_24 ** 2))**(1/2)))
  
  # angle between hip line and each shoulder
  a_11_23_24 = math.pi - math.acos((x_11_23 * x_23_24 + y_11_23 * y_23_24 + z_11_23 * z_23_24)/(((x_11_23 ** 2)+(y_11_23 ** 2)+(z_11_23 ** 2))**(1/2) * ((x_23_24 **2)+(y_23_24 ** 2)+(z_23_24 ** 2))**(1/2)))
  a_12_24_23 = math.acos((x_12_24 * x_23_24 + y_12_24 * y_23_24 + z_12_24 * z_23_24)/(((x_12_24 ** 2)+(y_12_24 ** 2)+(z_12_24 ** 2))**(1/2) * ((x_23_24 **2)+(y_23_24 ** 2)+(z_23_24 ** 2))**(1/2)))
  
  print("Angle between shoulder and hip: ", str(a_sh))
  print("Angle between hip line and left shoulder: ", str(a_11_23_24))
  print("Angle between hip line and right shoulder: ", str(a_12_24_23))
  print("========================")


  # Draw pose landmarks on the image.
  annotated_image = image.copy()
  # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
  # upper_body_only is set to True.
  mp_drawing.draw_landmarks(
      annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
  cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)


