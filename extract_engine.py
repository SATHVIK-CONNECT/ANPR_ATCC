import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import re
import tempfile
import base64
from groq import Groq

class Violation:
    # --- INITIALIZATION ---
    st.set_page_config(page_title="AI ANPR & Helmet Enforcement", layout="wide")
    st.title("ANPR & Helmet Violation Detection (Groq Powered)")

    # Initialize Groq - Replace with your actual key
    client = Groq(api_key="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")

    @st.cache_resource
    def load_yolo():
        # Load your custom YOLO model
        return YOLO("model3/best.pt") 

    yolo_model = load_yolo()

    # --- HELPER FUNCTIONS ---

    def clean_plate_text(text):
        """Removes noise and returns clean alpha-numeric text."""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def encode_image_to_base64(image_np):
        """Converts a numpy image crop to base64 for Groq API."""
        _, buffer = cv2.imencode('.jpg', image_np)
        return base64.b64encode(buffer).decode('utf-8')

    def extract_plate_with_groq(crop_np):
        """Sends the plate crop to Groq Vision for high-accuracy OCR."""
        base64_image = encode_image_to_base64(crop_np)
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract only the alpha-numeric license plate number from this image. Output only the characters and nothing else."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
                temperature=0,
            )
            return clean_plate_text(response.choices[0].message.content)
        except Exception as e:
            return f"ERROR: {str(e)}"

    def get_bike_status(bike_box, all_boxes, names):
        """Checks if a bike instance is associated with a helmet or a violation."""
        bx1, by1, bx2, by2 = map(int, bike_box.xyxy[0])
        has_helmet = False
        has_violation = False 

        for box in all_boxes:
            label = names[int(box.cls[0])].lower()
            ox_c = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            oy_c = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            
            if bx1 <= ox_c <= bx2 and by1 <= oy_c <= by2:
                if "no helmet" in label or "head" in label:
                    has_violation = True
                elif "helmet" in label:
                    has_helmet = True

        is_safe = has_helmet and not has_violation
        return (is_safe, "#28a745", "HELMET DETECTED") if is_safe else (False, "#dc3545", "VIOLATION: NO HELMET")

    # --- UI SIDEBAR ---
    with st.sidebar:
        st.header("Configuration")
        mode = "Image"
        conf_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.45)
        plate_padding = st.slider("Plate Crop Padding", 0, 50, 10)
        if mode == "Video":
            frame_skip = st.slider("Frame Skip", 1, 30, 10)

    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader(f"Upload {mode}", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        if mode == "Image":
            img_np = np.array(Image.open(uploaded_file).convert("RGB"))
            results = yolo_model.predict(img_np, conf=conf_threshold)
            names = yolo_model.names
            boxes = results[0].boxes

            # Visualize main detection
            st.image(results[0].plot(), caption="Detection Overview", use_container_width=True)

            # Separate detections
            bikes = [b for b in boxes if names[int(b.cls[0])].lower() in ["motorcycle", "bike"]]
            plates = [b for b in boxes if "plate" in names[int(b.cls[0])].lower()]

            for b_idx, bike in enumerate(bikes):
                is_safe, status_color, status_text = get_bike_status(bike, boxes, names)
                bx1, by1, bx2, by2 = map(int, bike.xyxy[0])
                
                # Find plates within this bike's bounding box
                bike_plates = []
                for p in plates:
                    px_c, py_c = (p.xyxy[0][0] + p.xyxy[0][2]) / 2, (p.xyxy[0][1] + p.xyxy[0][3]) / 2
                    if bx1 <= px_c <= bx2 and by1 <= py_c <= by2:
                        bike_plates.append(p)

                st.divider()
                st.markdown(f"### Bike {b_idx+1} | <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
                
                if bike_plates:
                    cols = st.columns(len(bike_plates))
                    for p_idx, p_box in enumerate(bike_plates):
                        with cols[p_idx]:
                            # Crop the plate
                            px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                            y_min, y_max = max(0, py1-plate_padding), min(img_np.shape[0], py2+plate_padding)
                            x_min, x_max = max(0, px1-plate_padding), min(img_np.shape[1], px2+plate_padding)
                            plate_crop = img_np[y_min:y_max, x_min:x_max]
                            
                            # Process with Groq
                            with st.spinner("Extracting Plate via Groq..."):
                                plate_text = extract_plate_with_groq(plate_crop)
                            
                            st.image(plate_crop, caption="Plate Crop")
                            st.markdown(f"""
                                <div style="background-color:{status_color}; padding:10px; border-radius:5px; text-align:center; color:white;">
                                    <h2 style="margin:0;">{plate_text}</h2>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No plate detected for this motorcycle.")

