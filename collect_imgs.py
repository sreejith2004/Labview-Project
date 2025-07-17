import os
import cv2

# -------------------------- CONFIG --------------------------
DATA_DIR = './data'
number_of_classes = 14
dataset_size = 100
camera_index = 0 # Change to 0 or 1 if using internal webcam

# -------------------------- SETUP --------------------------
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Cannot access camera. Check camera index or permissions.")
    exit()

# -------------------------- DATA COLLECTION --------------------------
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)
    print(f'\nCollecting data for class {j}')

    # Wait for user to press 'q' to start
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start...', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 27:  # ESC key
            print("Exiting early...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Capture dataset_size number of images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('frame', frame)
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

        if cv2.waitKey(1) == 27:  # ESC to exit
            print("Interrupted!")
            break

    print(f'Done collecting for class {j}')

cap.release()
cv2.destroyAllWindows()
print("\nDataset collection complete.")
