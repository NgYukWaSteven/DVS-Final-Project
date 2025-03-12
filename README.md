# DVS-Final-Project
Design for Visual Systems Final Project

#Please follow the following to setup
Download main.py from github

# Gesture Filter & Object Extraction Interface

## Project Overview

This project is an advanced gesture-controlled application built using OpenCV and MediaPipe. It integrates multiple interaction modes into a single application:

- **Filter Mode (Normal Mode):**  
  The application applies various visual filters (Color Map, Gamma Correction, Cartoon, Inverted Colors) to the live webcam feed in real time. You can adjust the filter parameters using an on-screen slider.

- **Settings Mode (Filter Selection & Adjustment):**  
  Toggled by a left-hand gesture (holding up only the thumb and pinky for 2 seconds), settings mode displays filter icons at the top of the screen. You can select a filter either using the right-hand relative cursor (driven by your hand motion) or directly by pinching with your right hand (or using your right index fingertip). Once selected, a slider appears at the bottom to adjust the filter’s intensity or gamma value.

- **Object Extraction & Interaction Mode:**  
  When you hold a steady left-hand pinch (using your index finger and thumb) for more than 3 seconds, the application extracts an object from the scene using GrabCut for precise segmentation. The extracted object is produced with a transparent background and is "frozen" in place—meaning it will not automatically follow your hand. After extraction, you can interact with the object by moving or scaling it using one or both hands.

- **Right-Hand Cursor:**  
  A relative cursor (initially placed at the screen center) is driven by the right hand’s movement. This cursor, along with direct pinch detection, is used to click on on-screen buttons.

---

## How to Use the Application

### Normal Mode (Filter Mode)

**Viewing:**  
By default, the application runs in normal mode, applying the currently selected visual filter to the live video feed.

**Filter Display:**  
The name of the active filter is shown in the upper left corner.

### Settings Mode (Filter Selection & Adjustment)

**Toggling Settings Mode:**  
Hold your left hand with only the thumb and pinky extended (while the index, middle, and ring fingers are down) for 2 seconds. This gesture toggles settings mode on or off.

**Filter Selection:**  
When settings mode is active, filter icons (for Default, Color, Gamma, Cartoon, and Invert) appear at the top of the screen.  
You can select a filter using the right-hand relative cursor (which is driven by your hand motion) or directly by pinching with your right hand (or using your right index fingertip).

**Adjusting Filter Parameters:**  
Once a filter is selected, a slider appears at the bottom of the screen. Use the cursor or a direct right-hand pinch to adjust the slider and change the filter’s intensity or gamma value.

### Object Extraction & Interaction Mode

**Extracting an Object:**  
With your left hand, perform a pinch (using the index finger and thumb) and hold it steadily for more than 3 seconds. A countdown will appear near your pinch to indicate when extraction will occur.  
The application uses GrabCut to precisely extract the object from the scene and produces an image with a transparent background.

**Object Interaction:**  
After extraction, the object is “frozen” in place (it will not follow your hand automatically). To interact with the object:  
- **Move:** Pinch inside the object’s area (its bounding box) with either hand. The object will follow the pinch’s movement.  
- **Scale:** If you pinch inside the object with both hands simultaneously, the object’s scale will change based on the distance between the two pinch centers.

**Exiting Object Mode:**  
Press the `x` key to exit object mode. The extracted object will disappear.

### Exiting the Application

Press the `q` key to quit the program.

---

## Evidence of Functionality

**Screenshots:**
- **Normal Mode:** A screenshot showing the live video feed with a visual filter applied.
- **Settings Mode:** A screenshot displaying the settings interface with filter icons and the parameter slider.
- **Object Extraction:** A screenshot of an extracted object overlaid on the video feed with a transparent background.

**Demo Video:**  
A short video demonstrating:
- Toggling settings mode using the left-hand gesture.
- Selecting and adjusting filters using the right-hand cursor and pinch.
- Extracting an object using a steady left-hand pinch and interacting with the object (moving and scaling).

---

## Evaluation

**Strengths:**
- **Versatile Interaction:**  
  Multiple interaction modes allow users to control filters and extract objects using natural hand gestures.
- **Accurate Gesture Recognition:**  
  MediaPipe provides robust hand landmark detection for reliable pinch and finger configuration recognition.
- **Advanced Object Extraction:**  
  The use of GrabCut enables precise segmentation of objects with transparent backgrounds.
- **Flexible Input Methods:**  
  Supports both a relative cursor and direct pinch clicks for enhanced usability.

**Limitations:**
- **Environmental Sensitivity:**  
  Varying lighting conditions and complex backgrounds may affect gesture recognition and object segmentation.
- **Hardware Performance:**  
  Real-time video processing and multiple interaction modes can be demanding on lower-end hardware, possibly causing latency.
- **Calibration:**  
  Thresholds for pinch detection and extraction stability might require tuning for different users and environments.
- **Robustness:**  
  Rapid hand movements or partial occlusions may sometimes lead to gesture misinterpretation.

---

## Conclusion

This project demonstrates a sophisticated integration of gesture recognition, real-time visual filtering, and precise object extraction using state-of-the-art computer vision techniques. It provides an intuitive, multi-mode interaction system that allows users to easily adjust filters and interact with extracted objects. Further refinements may be needed to enhance robustness in diverse environments.

---

*Team Members: [Your Name] and [Partner's Name]*  
*Date: [Submission Date]*
