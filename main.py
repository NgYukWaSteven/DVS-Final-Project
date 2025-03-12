import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ---------------------------
# Initialize MediaPipe Hands
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ---------------------------
# Global Variables for Settings Mode Toggle (Left Hand Gesture)
# ---------------------------
settings_mode = False
left_tp_gesture_start_time = None  # left thumb+pinky gesture (held for 2 sec)

# ---------------------------
# Filter Settings (for filter mode)
# ---------------------------
# Filters:
# 0: Default (no filter)
# 1: Color Map effect (COLORMAP_JET blend)
# 2: Gamma Correction (gamma adjustable)
# 3: Cartoon Effect (bilateral filter + edge mask)
# 4: Inverted Colors (blend inverted with original)
selected_filter = 0  # initially default
filter_params = {
    0: 0.0,   # not used
    1: 1.0,   # intensity for Color Map effect
    2: 0.31,  # slider value mapping to gamma (0.31 ~ gamma=1.0)
    3: 1.0,   # intensity for Cartoon effect
    4: 1.0    # blend ratio for Inverted effect
}
filter_labels = {
    0: "Default",
    1: "Color",
    2: "Gamma",
    3: "Cartoon",
    4: "Invert"
}
slider_active = False
slider_region = None
last_settings_selection_time = 0
settings_selection_debounce_delay = 0.3  # seconds

# ---------------------------
# Cursor Variables (Right Hand Relative Movement)
# ---------------------------
cursor_pos = None            # global cursor position (pixels)
prev_right_centroid = None   # previous centroid of right hand (pixels)
last_click_time = 0          # for visual "click" feedback

# ---------------------------
# Object Extraction & Interaction Variables
# ---------------------------
object_mode = False          # if True, an object has been extracted
object_img = None            # extracted object image (RGBA, transparent background)
object_pos = None            # fixed center position (pixels) where object is placed
object_scale = 1.0           # scaling factor for the object

# Extraction variables
extraction_in_progress = False
extraction_start_time = None
extraction_initial_pinch_center = None
extraction_stability_threshold = 30    # allowed movement in pixels during extraction
extraction_duration_threshold = 3.0      # seconds required for extraction
extraction_pinch_threshold = 80          # relaxed threshold for larger objects

# Object interaction variables
object_grabbed_left = False
object_grabbed_right = False
left_grab_offset = None
right_grab_offset = None
object_initial_two_pinch_distance = None
object_initial_scale = None

# ---------------------------
# Helper Functions
# ---------------------------
def get_finger_states(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    index_extended = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    ring_extended = lm[16].y < lm[14].y
    pinky_extended = lm[20].y < lm[18].y
    if hand_label == "Right":
        thumb_extended = lm[4].x < lm[3].x
    else:
        thumb_extended = lm[4].x > lm[3].x
    return (thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended)

def calc_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def is_pinch(hand_landmarks, width, height, pinch_threshold=40):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_pos = (int(index_tip.x * width), int(index_tip.y * height))
    thumb_pos = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    distance = math.hypot(index_pos[0] - thumb_pos[0], index_pos[1] - thumb_pos[1])
    return (distance < pinch_threshold, index_pos, thumb_pos)

def draw_slider(frame, region, value, label):
    x, y, w, h = region
    cv2.rectangle(frame, (x, y), (x+w, y+h), (200,200,200), 2)
    knob_x = int(x + value * w)
    cv2.circle(frame, (knob_x, y + h//2), h//2, (0,0,255), -1)
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def apply_filter(frame, filter_index, param):
    if filter_index == 0:
        return frame
    elif filter_index == 1:
        colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, param, frame, 1-param, 0)
    elif filter_index == 2:
        gamma = 0.1 + param * 2.9
        invGamma = 1.0 / gamma
        table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(frame, table)
    elif filter_index == 3:
        cartoon = cv2.bilateralFilter(frame, 9, 75, 75)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(cartoon, param, edges, 1-param, 0)
    elif filter_index == 4:
        inverted = cv2.bitwise_not(frame)
        return cv2.addWeighted(inverted, param, frame, 1-param, 0)
    else:
        return frame

def extract_object_precise(frame, center, size=150):
    # Use GrabCut for a more precise extraction with transparency.
    h, w = frame.shape[:2]
    x, y = int(center[0]), int(center[1])
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    roi = frame[y1:y2, x1:x2].copy()
    mask = np.zeros(roi.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (5, 5, roi.shape[1]-10, roi.shape[0]-10)
    cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    extracted = roi * mask2[:,:,np.newaxis]
    # Create an alpha channel from the mask (transparent where mask2==0)
    alpha = mask2 * 255
    extracted_rgba = cv2.cvtColor(extracted, cv2.COLOR_BGR2BGRA)
    extracted_rgba[:,:,3] = alpha
    return extracted_rgba

def overlay_image_alpha(background, overlay_img, pos):
    # Composite overlay_img (with alpha channel) onto background at pos.
    x, y = pos
    overlay_h, overlay_w = overlay_img.shape[:2]
    # Ensure ROI is within background bounds
    if x < 0 or y < 0 or x+overlay_w > background.shape[1] or y+overlay_h > background.shape[0]:
        return background
    roi = background[y:y+overlay_h, x:x+overlay_w]
    overlay_rgb = overlay_img[:,:,:3]
    alpha = overlay_img[:,:,3] / 255.0
    # Blend: out = alpha*overlay + (1-alpha)*roi
    blended = (alpha[..., np.newaxis] * overlay_rgb + (1 - alpha[..., np.newaxis]) * roi).astype(np.uint8)
    background[y:y+overlay_h, x:x+overlay_w] = blended
    return background

# ---------------------------
# Set Up Video Capture
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Gesture Filter Interface", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Filter Interface", 1280, 720)

# ---------------------------
# Main Loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror image for natural interaction
    h, w, _ = frame.shape
    overlay = frame.copy()
    current_time = time.time()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    hand_results = hands.process(img_rgb)
    left_hand_landmarks = None
    right_hand_landmarks = None
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_handedness, hand_landmarks in zip(hand_results.multi_handedness, hand_results.multi_hand_landmarks):
            label = hand_handedness.classification[0].label  # "Left" or "Right"
            if label == "Left":
                left_hand_landmarks = hand_landmarks
            else:
                right_hand_landmarks = hand_landmarks
            mp_draw.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------------------------
    # Left-Hand Settings Mode Toggle (if not in object mode)
    # ---------------------------
    if not object_mode and left_hand_landmarks is not None:
        left_states = get_finger_states(left_hand_landmarks, "Left")
        # Desired gesture: thumb and pinky extended; index, middle, ring not extended.
        if left_states[0] and (not left_states[1]) and (not left_states[2]) and (not left_states[3]) and left_states[4]:
            if left_tp_gesture_start_time is None:
                left_tp_gesture_start_time = current_time
            else:
                if current_time - left_tp_gesture_start_time >= 2.0:
                    settings_mode = not settings_mode
                    slider_active = False  # reset slider on toggle
                    left_tp_gesture_start_time = None
        else:
            left_tp_gesture_start_time = None

    # ---------------------------
    # Right Hand Cursor (Relative Movement)
    # ---------------------------
    if right_hand_landmarks is not None:
        xs = [lm.x for lm in right_hand_landmarks.landmark]
        ys = [lm.y for lm in right_hand_landmarks.landmark]
        current_centroid = (np.mean(xs)*w, np.mean(ys)*h)
        if prev_right_centroid is None or cursor_pos is None:
            cursor_pos = (w//2, h//2)
            prev_right_centroid = current_centroid
        else:
            dx = current_centroid[0] - prev_right_centroid[0]
            dy = current_centroid[1] - prev_right_centroid[1]
            sensitivity = 2.0
            new_x = int(cursor_pos[0] + dx * sensitivity)
            new_y = int(cursor_pos[1] + dy * sensitivity)
            cursor_pos = (max(0, min(w, new_x)), max(0, min(h, new_y)))
            prev_right_centroid = current_centroid
    else:
        cursor_pos = None
        prev_right_centroid = None

    # Right-hand pinch for click detection
    right_click = False
    right_pinch_center = None
    if right_hand_landmarks is not None:
        pinch_detected, index_pos, thumb_pos = is_pinch(right_hand_landmarks, w, h)
        if pinch_detected:
            last_click_time = current_time
            right_click = True
            right_pinch_center = index_pos
    # Also record the right index fingertip (for direct click)
    right_index = None
    if right_hand_landmarks is not None:
        right_index = (int(right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w),
                       int(right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*h))
    # Combine potential click positions: pinch center, index fingertip, and cursor.
    click_positions = []
    if right_pinch_center is not None:
        click_positions.append(right_pinch_center)
    if right_index is not None:
        click_positions.append(right_index)
    if cursor_pos is not None:
        click_positions.append(cursor_pos)
    # Draw the right-hand cursor if available.
    if cursor_pos is not None:
        color = (0,0,255) if current_time - last_click_time < 0.3 else (255,0,0)
        cv2.circle(overlay, cursor_pos, 10, color, -1)

    # ---------------------------
    # Object Extraction (Left Hand Pinch)
    # ---------------------------
    if not object_mode and left_hand_landmarks is not None:
        left_pinch, left_index, left_thumb = is_pinch(left_hand_landmarks, w, h, pinch_threshold=extraction_pinch_threshold)
        if left_pinch:
            current_left_pinch_center = ((left_index[0]+left_thumb[0])//2,
                                         (left_index[1]+left_thumb[1])//2)
            # Display countdown near pinch
            if extraction_in_progress:
                elapsed = current_time - extraction_start_time
                remaining = max(0, int(extraction_duration_threshold - elapsed))
                cv2.putText(overlay, f"Extract in {remaining} sec",
                            (current_left_pinch_center[0]-50, current_left_pinch_center[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if not extraction_in_progress:
                extraction_in_progress = True
                extraction_start_time = current_time
                extraction_initial_pinch_center = current_left_pinch_center
            else:
                if calc_distance(current_left_pinch_center, extraction_initial_pinch_center) < extraction_stability_threshold:
                    if current_time - extraction_start_time >= extraction_duration_threshold:
                        # Extract object using GrabCut for precise segmentation with transparency.
                        object_img = extract_object_precise(frame, current_left_pinch_center, size=150)
                        object_mode = True
                        object_pos = current_left_pinch_center  # Freeze object's position
                        object_scale = 1.0
                        extraction_in_progress = False
                        extraction_start_time = None
                        extraction_initial_pinch_center = None
                else:
                    extraction_in_progress = False
                    extraction_start_time = None
                    extraction_initial_pinch_center = None
        else:
            extraction_in_progress = False
            extraction_start_time = None
            extraction_initial_pinch_center = None

    # ---------------------------
    # Object Interaction Mode
    # ---------------------------
    if object_mode:
        # Compute the object's bounding box based on its RGBA image size and scale.
        if object_img is not None and object_pos is not None:
            obj_h, obj_w = object_img.shape[:2]
            half_w = (obj_w * object_scale) / 2
            half_h = (obj_h * object_scale) / 2
            obj_box = (object_pos[0] - half_w, object_pos[1] - half_h,
                       object_pos[0] + half_w, object_pos[1] + half_h)
            # (Do not draw the green box if you prefer not to see it)
            # cv2.rectangle(overlay, (int(obj_box[0]), int(obj_box[1])), (int(obj_box[2]), int(obj_box[3])), (0,255,0), 2)
            # Instead, we overlay the object with transparency.
            new_w = int(obj_w * object_scale)
            new_h = int(obj_h * object_scale)
            resized_obj = cv2.resize(object_img, (new_w, new_h))
            top_left = (int(object_pos[0] - new_w/2), int(object_pos[1] - new_h/2))
            overlay = overlay_image_alpha(overlay, resized_obj, top_left)
        # Determine if either hand is “grabbing” the object (only if pinch occurs within the object's box).
        left_obj_grab = False
        left_obj_center = None
        if left_hand_landmarks is not None:
            lp, li, lt = is_pinch(left_hand_landmarks, w, h, pinch_threshold=extraction_pinch_threshold)
            if lp:
                left_obj_center = ((li[0]+lt[0])//2, (li[1]+lt[1])//2)
                if obj_box is not None and (obj_box[0] <= left_obj_center[0] <= obj_box[2] and
                                             obj_box[1] <= left_obj_center[1] <= obj_box[3]):
                    left_obj_grab = True
        right_obj_grab = False
        right_obj_center = None
        if right_hand_landmarks is not None:
            rp, ri, rt = is_pinch(right_hand_landmarks, w, h)
            if rp:
                right_obj_center = ((ri[0]+rt[0])//2, (ri[1]+rt[1])//2)
                if obj_box is not None and (obj_box[0] <= right_obj_center[0] <= obj_box[2] and
                                             obj_box[1] <= right_obj_center[1] <= obj_box[3]):
                    right_obj_grab = True
        # Update object position/scale only if the pinch(s) occur inside the object's bounding box.
        if left_obj_grab and right_obj_grab:
            new_center = ((left_obj_center[0] + right_obj_center[0])//2,
                          (left_obj_center[1] + right_obj_center[1])//2)
            object_pos = new_center
            current_distance = calc_distance(left_obj_center, right_obj_center)
            if object_initial_two_pinch_distance is None:
                object_initial_two_pinch_distance = current_distance
                object_initial_scale = object_scale
            else:
                scale_factor = current_distance / object_initial_two_pinch_distance
                object_scale = object_initial_scale * scale_factor
        elif left_obj_grab and not right_obj_grab:
            if not object_grabbed_left:
                object_grabbed_left = True
                left_grab_offset = (object_pos[0] - left_obj_center[0], object_pos[1] - left_obj_center[1])
            else:
                object_pos = (left_obj_center[0] + left_grab_offset[0], left_obj_center[1] + left_grab_offset[1])
            object_initial_two_pinch_distance = None
            object_initial_scale = None
            object_grabbed_right = False
            right_grab_offset = None
        elif right_obj_grab and not left_obj_grab:
            if not object_grabbed_right:
                object_grabbed_right = True
                right_grab_offset = (object_pos[0] - right_obj_center[0], object_pos[1] - right_obj_center[1])
            else:
                object_pos = (right_obj_center[0] + right_grab_offset[0], right_obj_center[1] + right_grab_offset[1])
            object_initial_two_pinch_distance = None
            object_initial_scale = None
            object_grabbed_left = False
            left_grab_offset = None
        else:
            object_grabbed_left = False
            object_grabbed_right = False
            left_grab_offset = None
            right_grab_offset = None
            object_initial_two_pinch_distance = None
            object_initial_scale = None
        if left_obj_grab or right_obj_grab:
            cv2.putText(overlay, "OBJECT MODE (press x to exit)", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ---------------------------
    # Settings Mode (Filter Selection)
    # ---------------------------
    if settings_mode and not object_mode:
        icon_radius = 30
        icon_spacing = 100
        start_x = w//2 - 2*icon_spacing
        icon_y = 50
        icon_positions = {
            0: (start_x + 0*icon_spacing, icon_y),
            1: (start_x + 1*icon_spacing, icon_y),
            2: (start_x + 2*icon_spacing, icon_y),
            3: (start_x + 3*icon_spacing, icon_y),
            4: (start_x + 4*icon_spacing, icon_y)
        }
        for i in range(5):
            pos = icon_positions[i]
            cv2.circle(overlay, pos, icon_radius, (0,255,255), -1)
            cv2.putText(overlay, filter_labels[i], (pos[0]-40, pos[1]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(overlay, "SETTINGS MODE", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        # Gather multiple click positions: right-hand pinch, right index fingertip, and the cursor.
        click_positions = []
        if right_pinch_center is not None:
            click_positions.append(right_pinch_center)
        if right_hand_landmarks is not None:
            right_index = (int(right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w),
                           int(right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*h))
            click_positions.append(right_index)
        if cursor_pos is not None:
            click_positions.append(cursor_pos)
        # Check each icon: if any click position is within icon radius, register the click.
        for i in range(5):
            pos = icon_positions[i]
            for cp in click_positions:
                if cp is not None and calc_distance(cp, pos) < 40 and (current_time - last_settings_selection_time > settings_selection_debounce_delay):
                    selected_filter = i
                    slider_active = True
                    last_settings_selection_time = current_time
                    break
        if slider_active:
            slider_x = w//2 - 150
            slider_y = h - 100
            slider_width = 300
            slider_height = 30
            slider_region = (slider_x, slider_y, slider_width, slider_height)
            if selected_filter == 0:
                slider_label = "No Filter (N/A)"
            elif selected_filter == 1:
                slider_label = "Color Map Intensity"
            elif selected_filter == 2:
                slider_label = "Gamma (0.1 to 3.0)"
            elif selected_filter == 3:
                slider_label = "Cartoon Intensity"
            elif selected_filter == 4:
                slider_label = "Invert Blend Ratio"
            current_value = filter_params[selected_filter]
            draw_slider(overlay, slider_region, current_value, slider_label)
            for cp in click_positions:
                if cp is not None and right_click:
                    if slider_x <= cp[0] <= slider_x+slider_width and slider_y <= cp[1] <= slider_y+slider_height:
                        new_val = (cp[0] - slider_x) / slider_width
                        filter_params[selected_filter] = max(0.0, min(1.0, new_val))

    # ---------------------------
    # Normal Mode (Filter Application)
    # ---------------------------
    if not settings_mode and not object_mode:
        overlay = apply_filter(overlay, selected_filter, filter_params[selected_filter])
        cv2.putText(overlay, f"Filter: {filter_labels[selected_filter]}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Gesture Filter Interface", overlay)
    key = cv2.waitKey(1)
    # Press 'x' to exit object mode, 'q' to quit.
    if key == ord('x'):
        object_mode = False
        object_img = None
        object_pos = None
        object_scale = 1.0
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
