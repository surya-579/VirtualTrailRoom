import cv2

goggles_img = cv2.imread('./eyecaps.png', -1)
cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        goggles_resized = cv2.resize(goggles_img, (w, int(goggles_img.shape[0] * (w / goggles_img.shape[1]))))
        y_offset = y + int(h / 4)
        x_offset = x
        
        y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + goggles_resized.shape[0])
        x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + goggles_resized.shape[1])
        
        y1_g, y2_g = max(0, -y_offset), min(goggles_resized.shape[0], frame.shape[0] - y_offset)
        x1_g, x2_g = max(0, -x_offset), min(goggles_resized.shape[1], frame.shape[1] - x_offset)
        
        if goggles_resized.shape[2] == 4:
            alpha_goggles = goggles_resized[y1_g:y2_g, x1_g:x2_g, 3] / 255.0
            alpha_frame = 1.0 - alpha_goggles
            
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_goggles * goggles_resized[y1_g:y2_g, x1_g:x2_g, c] +
                                           alpha_frame * frame[y1:y2, x1:x2, c])
    
    cv2.imshow('Virtual Try-On', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
