import cv2
import mediapipe as mp
import os
import json
from pathlib import Path

base_dir = Path(__file__).parent
config_path = base_dir / 'config.json'
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

train_dir = base_dir / r'Train_Alphabet'
test_dir = base_dir / r'Test_Alphabet'
processed_train_dir = base_dir / r'processed_Train_Alphabet'
processed_test_dir = base_dir / r'processed_Test_Alphabet'
num_train_images = config['num_train_images']
num_test_images = config['num_test_images']

print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")
print(f"Processed train directory: {processed_train_dir}")
print(f"Processed test directory: {processed_test_dir}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def create_directories(base_dir, classes):
    for class_name in classes:
        os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

def process_images(input_dir, output_dir, target_size):
    create_directories(output_dir, os.listdir(input_dir))
    for class_name in os.listdir(input_dir):
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        images = os.listdir(class_input_dir)
        for i, image_name in enumerate(images):
            if i >= target_size:
                break 
            image_path = os.path.join(class_input_dir, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            output_path = os.path.join(class_output_dir, image_name)
            cv2.imwrite(output_path, image)
            print(f"Processed image saved: {output_path}") 

if not train_dir.exists():
    raise FileNotFoundError(f"Directory '{train_dir}' does not exist.")
if not test_dir.exists():
    raise FileNotFoundError(f"Directory '{test_dir}' does not exist.")

create_directories(processed_train_dir, os.listdir(train_dir))
create_directories(processed_test_dir, os.listdir(test_dir))

# Process images
process_images(train_dir, processed_train_dir, num_train_images)
process_images(test_dir, processed_test_dir, num_test_images)

print("Images processed with Mediapipe and saved to processed directories.")
