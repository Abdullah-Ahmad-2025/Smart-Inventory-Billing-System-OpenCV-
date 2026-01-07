import cv2
import numpy as np
import time

#  Configuration : Ready for FIRST GO

VIDEO_PATH = 0  # WEBCAM
OUTPUT_VIDEO = 'inventory_output.mp4'
LOG_FILE = 'inventory_log.txt'

# COLOR TAGS WITH THEIR ASSOCIATED PRICES
COLORS = {
    "RED":  150,
    "BLUE":  80,
    "GREEN":  120
}


# Warping points if needed (source/dest for perspective correction)

WARP_ENABLED = False  # VIDEO IS NOT ANGLED, SO FALSE
SRC_POINTS = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
DST_POINTS = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

# Initializing video capture and writer
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found") # HANDLING WHEN NO WEBCAM DETECTED / EXIST
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Clear log file

with open(LOG_FILE, 'w') as f:
    f.write("Timestamp, Item Count\n")

frame_count = 0
start_time = time.time()

red_total = 0
blue_total = 0
green_total = 0

previous_red = False
previous_blue = False
previous_green = False

red_items = 0
blue_items = 0
green_items = 0


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    #  CONTRAST enhancement

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))

    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # For color preservation

    # Used enhanced for detection
    hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)

    scan_y1, scan_y2 = 300, 360 # A DETECTION STRIP
    roi = hsv[scan_y1:scan_y2, :] # RESTRICTING REGION OF INTEREST

    cv2.line(frame, (0, scan_y1), (width, scan_y1), (0, 255, 0), 2)
    cv2.line(frame, (0, scan_y2), (width, scan_y2), (0, 255, 0), 2)
    cv2.putText(frame, "SCAN ZONE", (10, scan_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Color thresholding and contour detection MASKS
    maskR1 = cv2.inRange(roi, (0, 90, 120), (8, 255, 255))
    maskR2 = cv2.inRange(roi, (170, 90, 120), (179, 255, 255))

    maskR = maskR1 | maskR2

    maskB = cv2.inRange(roi, (85, 110, 120), (105, 255, 255))

    maskG = cv2.inRange(roi, (25, 70, 110), (55, 255, 255))

    contoursR, _ = cv2.findContours(maskR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursB, _ = cv2.findContours(maskB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursG, _ = cv2.findContours(maskG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    red_present = False


    for cnt in contoursR:

        area = cv2.contourArea(cnt)

        if area > 800:  # Filtering small noise and tackling only count a single time

            red_present = True

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                frame,
                (x, y + scan_y1),
                (x + w, y + h + scan_y1),
                (255, 0, 0),
                2
            )
            break

    if red_present and not previous_red :
        red_total = red_total + 1
        red_items += COLORS["RED"]




    blue_present = False

    for cnt in contoursB:

        area = cv2.contourArea(cnt)


        if area > 800:  # Filter small noise

            blue_present = True
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                frame,
                (x, y + scan_y1),
                (x + w, y + h + scan_y1),
                (255, 0, 0),
                2
            )

            break

    if blue_present and not previous_blue :
        blue_total = blue_total + 1
        blue_items += COLORS["BLUE"]


    green_present = False

    for cnt in contoursG:

        area = cv2.contourArea(cnt)


        if area > 800:  # Filter small noise

            green_present = True
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(
                frame,
                (x, y + scan_y1),
                (x + w, y + h + scan_y1),
                (255, 0, 0),
                2
            )
            break

    if green_present and not previous_green :
        green_total = green_total + 1
        green_items += COLORS["GREEN"]

    previous_red = red_present
    previous_blue = blue_present
    previous_green = green_present

    # Warping (apply to frame if enabled)
    if WARP_ENABLED:

        matrix = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)
        frame = cv2.warpPerspective(frame, matrix, (width, height))

    # Overlay text and write output
    cv2.putText(frame, f'RED: {red_total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 0, 0), 2)
    cv2.putText(frame, f'BLUE: {blue_total}', (10, 65),cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 0, 0), 2)
    cv2.putText(frame, f'GREEN: {green_total}', (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 0, 0), 2)

    out.write(frame)

    # Log every 30 frames (or adjust)
    if frame_count % 30 == 0:

        elapsed = time.time() - start_time

        with open(LOG_FILE, 'a') as f:
            f.write(f'{elapsed:.2f}, RED, {red_total}, Rs {red_items}\n')
            f.write(f'{elapsed:.2f}, BLUE, {blue_total}, Rs {blue_items}\n')
            f.write(f'{elapsed:.2f}, GREEN, {green_total}, Rs {green_items}\n')

    frame_count += 1

    # Display
    cv2.imshow('Inventory Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n")
print("RED TOTAL : ",red_items, "Rs.")
print("BLUE TOTAL : ",blue_items, "Rs.")
print("GREEN TOTAL : ",green_items, "Rs.")
print("\n")

print("NET TOTAL : ",(red_items + blue_items + green_items), "Rs.")
print("\n")


# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output: {OUTPUT_VIDEO}, Log: {LOG_FILE}")