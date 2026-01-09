import numpy as np
import tflite_runtime.interpreter as tflite
import time
from picamera2 import Picamera2
from libcamera import controls
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import math

# --- INITIAL SETUP ---
factory = PiGPIOFactory()

# Define tilt and pan servos on specified GPIO pins
servo_tilt = Servo(17, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
servo_pan = Servo(27, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# --- ANGLE CONSTANTS ---
CENTER_POS = 90
WET_POS = 45
DRY_POS = 135

# --- CONFIGURATION ---
MODEL_PATH = 'model.tflite'
LABEL_PATH = 'labels.txt'

# --- SERVO HELPER FUNCTIONS ---
def deg_to_val(deg):
    return (deg / 90) - 1

def sine_sweep(servo, start_deg, end_deg, steps=200, delay=0.005):
    start_val = deg_to_val(start_deg)
    end_val = deg_to_val(end_deg)
    delta_val = end_val - start_val

    for i in range(steps + 1):
        t = i / steps
        eased_val = math.sin(math.pi * (t - 0.5))
        current_val = start_val + (delta_val * ((eased_val + 1) / 2))
        servo.value = current_val
        time.sleep(delay)

def run_deposit_sequence(target_angle):
    print(f"1. Panning to target angle ({target_angle} deg)")
    sine_sweep(servo_pan, CENTER_POS, target_angle)
    
    time.sleep(0.5)
    
    print("2. Tilting to drop waste...")
    sine_sweep(servo_tilt, CENTER_POS, 0)
    time.sleep(1)
    sine_sweep(servo_tilt, 0, CENTER_POS)
    
    print(f"3. Returning Pan to Center ({CENTER_POS} deg)")
    sine_sweep(servo_pan, target_angle, CENTER_POS)
    print("Deposit sequence complete.")

# --- APPLICATION INITIALIZATION ---
try:
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: Label file not found at {LABEL_PATH}. Check your path.")
    exit()

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
_, input_height, input_width, _ = input_details[0]['shape']

print("Initializing Pi Camera...")
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (input_width, input_height), "format": "RGB888"})
picam2.configure(config)
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
time.sleep(2)
print(f"Camera ready. Model input size: {input_width}x{input_height}.")

# Initial servo position
servo_tilt.value = deg_to_val(CENTER_POS)
servo_pan.value = deg_to_val(CENTER_POS)

# --- CLASSIFICATION CORE ---
def capture_and_classify():
    print("\n--- Starting Classification Cycle ---")
    print("Capturing image...")
    
    img_array = picam2.capture_array()
    
    input_data = np.array(img_array, dtype=np.float32)
    input_data = input_data / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    probabilities = output_data[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = labels[predicted_index]
    confidence = probabilities[predicted_index]
    
    return predicted_class, confidence

def trigger_actuator(waste_type):
    print(f"\n**Action:** Directing waste to the **{waste_type.upper()}** bin.")
    
    if waste_type.lower() == "wet":
        run_deposit_sequence(WET_POS)
    elif waste_type.lower() == "dry":
        run_deposit_sequence(DRY_POS)
    elif waste_type.lower() == "mixed":
        print("No pan required (Mixed Waste). Tilting to drop...")
        sine_sweep(servo_tilt, CENTER_POS, 0)
        time.sleep(1)
        sine_sweep(servo_tilt, 0, CENTER_POS)
    else:
        print("ERROR: Unknown classification result. Staying at center.")
        time.sleep(0.5)

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    try:
        print("\n\n*** READY ***")
        print("Press [ENTER] on the keyboard to CAPTURE and CLASSIFY waste. Press Ctrl+C to exit.")

        while True:
            # === WAIT FOR ENTER KEY PRESS ===
            input("\n[Awaiting input...] Press ENTER to continue sorting. ") 

            print("\n=============================================")
            
            predicted_class, confidence = capture_and_classify()
            
            print("\n--- CLASSIFICATION RESULT ---")
            print(f"Predicted Class: **{predicted_class}**")
            print(f"Confidence: {confidence:.2f}")
            
            trigger_actuator(predicted_class)
            
    except KeyboardInterrupt:
        print("\n[INFO] Program stopped by user (Ctrl+C).")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    finally:
        picam2.stop()
        servo_tilt.detach()
        servo_pan.detach()
        print("\nCamera stopped and servos detached. System resources released. Exiting.")