import cv2

cam_index = 0  # 换成你扫描出的可用 index

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Still cannot open camera, check if other apps are using it.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("NEXIGO Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
