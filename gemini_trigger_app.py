import os
import time
from google import genai
from google.genai.errors import APIError
from picamera2 import Picamera2
import numpy as np
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import math

# --- 1. HARDWARE SETUP & CONSTANTS ---
factory = PiGPIOFactory()
# Define tilt and pan servos on specified GPIO pins (17 and 27)
servo_tilt = Servo(17, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
servo_pan = Servo(27, pin_factory=factory, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

CENTER_POS = 90
WET_POS = 135  # Assuming Wet uses the dedicated 135 position (tilt-only)
DRY_POS = 45   # Using 45 as the Dry bin pan position for the requested sequence
MIXED_HAZ_POS = 135 # Mixed/Hazardous uses 135 (same as Dry)

API_KEY = os.environ.get("GEMINI_API_KEY")
CLASSIFICATION_MODEL = 'gemini-2.5-flash'
VALID_CATEGORIES = ["wet", "dry", "mixed", "electronic", "hazardous"] 
CAPTURE_PATH = os.path.join(os.getcwd(), 'capture_temp.jpg')

# --- 2. SERVO HELPER FUNCTIONS ---
def deg_to_val(deg):
    """Maps a degree value (0-180) to the Servo value range (-1.0 to +1.0)."""
    return (deg / 90) - 1

def sine_sweep(servo, start_deg, end_deg, steps=200, delay=0.005):
    """Move servo smoothly from start_deg to end_deg using sine easing."""
    start_val = deg_to_val(start_deg)
    end_val = deg_to_val(end_deg)
    delta_val = end_val - start_val
    for i in range(steps + 1):
        t = i / steps
        eased_val = math.sin(math.pi * (t - 0.5))
        current_val = start_val + (delta_val * ((eased_val + 1) / 2))
        servo.value = current_val
        time.sleep(delay)

def run_drop_tilt_cycle(tilt_pos, return_pos=CENTER_POS):
    """Performs the 90 -> X -> 90 tilt cycle for dropping waste."""
    sine_sweep(servo_tilt, return_pos, tilt_pos) # Go down/up
    time.sleep(1) 
    sine_sweep(servo_tilt, tilt_pos, return_pos) # Return to neutral

def run_pan_tilt_sequence(pan_target, tilt_target):
    """Executes the full Pan-Tilt-Drop sequence for Dry/Mixed/Hazardous."""
    
    # 1. Pan to the target compartment
    print(f"1. Panning to target angle ({pan_target} deg)")
    sine_sweep(servo_pan, CENTER_POS, pan_target)
    time.sleep(0.5) 
    
    # 2. Tilt cycle (90 -> 180 -> 90)
    print("2. Tilting fully to 180 degrees...")
    sine_sweep(servo_tilt, CENTER_POS, 180) # 90 -> 180 (Drop)
    time.sleep(1) 
    sine_sweep(servo_tilt, 180, CENTER_POS) # 180 -> 90 (Close/Neutral)
    
    # 3. Pan back to Center
    print(f"3. Returning Pan to Center ({CENTER_POS} deg)")
    sine_sweep(servo_pan, pan_target, CENTER_POS)
    print("Deposit sequence complete.")

# --- 3. INITIALIZATION (Camera & Servos) ---
print("Initializing Pi Camera...")
picam2 = Picamera2()
capture_width, capture_height = 640, 480 
config = picam2.create_still_configuration(main={"size": (capture_width, capture_height), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2) 
print(f"Camera ready at {capture_width}x{capture_height}.")

# Initial servo position (Center/Neutral)
servo_tilt.value = deg_to_val(CENTER_POS)
servo_pan.value = deg_to_val(CENTER_POS)

# --- 4. GEMINI API LOGIC ---
def capture_image_from_pi_cam(path):
    """Captures and saves image."""
    try:
        picam2.capture_file(path) 
        return True
    except Exception as e:
        print(f"Error capturing image: {e}")
        return False

def classify_with_gemini(path):
    """Sends the captured image as bytes to Gemini and returns the classification."""
    if not os.path.exists(path): return "error: no image"
    if not API_KEY: return "error: API key not set"
        
    try:
        client = genai.Client(api_key=API_KEY)
        
        # Read the image as bytes (The FIX for the Part.from_uri() bug)
        with open(path, 'rb') as f:
            image_bytes = f.read()

        image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
        
        # UPDATED SYSTEM INSTRUCTION: Constrains output to the three physical destinations
        system_instruction = (
            "You are an expert waste classification AI. Classify the item into one and only one of the following THREE categories: "
            "'wet' (food/organic), 'dry' (clean paper/plastic), or 'mixed' (contaminated, electronic, or hazardous waste). "
            "Respond with ONLY the category name, in lowercase."
        )

        response = client.models.generate_content(
            model=CLASSIFICATION_MODEL,
            contents=[image_part, "Classify the waste item according to the rules."],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1, 
            )
        )

        classification_result = response.text.strip().lower()
        return classification_result if classification_result in ['wet', 'dry', 'mixed'] else "unknown"

    except APIError as e: return f"error: api {e}"
    except Exception as e: return f"error: general {e}"


def trigger_actuator(classification):
    """Routes the classification result to the specific motor sequence."""
    print(f"\n**Action:** Directing waste to the **{classification.upper()}** bin.")
    
    if classification == 'wet':
        # WET WASTE: TILT ONLY (90 -> 0 -> 90)
        print("WET WASTE DETECTED: Running tilt-only drop sequence.")
        run_drop_tilt_cycle(tilt_pos=0)
        
    elif classification == 'dry':
        # DRY WASTE: PAN (90->45), TILT (90->180->90), PAN (45->90)
        print(f"DRY WASTE DETECTED: Running custom pan and full tilt drop sequence.")
        run_pan_tilt_sequence(pan_target=DRY_POS, tilt_target=180)

    elif classification == 'mixed':
        # MIXED WASTE: PAN (90->135), TILT (90->180->90), PAN (135->90)
        print("MIXED WASTE DETECTED: Running custom pan and full tilt drop sequence.")
        run_pan_tilt_sequence(pan_target=MIXED_HAZ_POS, tilt_target=180)
        
    else:
        print(f"Actuator ignored. Classification error or unknown result: {classification}")


# --- 5. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    try:
        print("\n\n*** WASTE CLASSIFIER READY ***")
        
        while True:
            # === WAIT FOR ENTER KEY PRESS (The Trigger) ===
            input("\n[Awaiting trigger] Press ENTER to capture and classify (Ctrl+C to exit).") 

            # 1. CAPTURE
            if not capture_image_from_pi_cam(CAPTURE_PATH): continue

            # 2. CLASSIFY
            print("Analyzing the image...")
            classification = classify_with_gemini(CAPTURE_PATH)
            
            # 3. OUTPUT & ACTUATE
            print("\n=============================================")
            print(f"WASTE CLASSIFICATION: **{classification.upper()}**")
            
            # Trigger the specific actuator sequence based on the final result
            trigger_actuator(classification)
            
            print("=============================================")
            
    except KeyboardInterrupt:
        print("\n[INFO] Program stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        picam2.stop()
        servo_tilt.detach()
        servo_pan.detach()
        print("Camera stopped and servos detached. System resources released. Exiting.")
