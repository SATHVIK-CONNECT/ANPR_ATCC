from ultralytics import YOLO
import cv2
import os
import math
import easyocr
import numpy as np

class ViolationEngine:
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf 
        
        self.helmet_conf = 0.80
        self.mobile_conf = 0.50
        self.rider_conf = 0.60  
        
        os.makedirs("saved_violations", exist_ok=True)
        # Store TRACK IDs here (e.g., 1, 2, 5) to remember distinct vehicles
        self.processed_violation_ids = set()
        
        print("Initializing EasyOCR...") 
        self.reader = easyocr.Reader(['en'], gpu=True) 

    def reset(self):
        self.processed_violation_ids.clear()

    # Helper to link a violation box (Helmet/Mobile) to a Rider Box
    def assign_violation_to_rider(self, violation_box, rider_data):
        # rider_data is a list of dicts: [{'box':..., 'id':...}, ...]
        vx1, vy1, vx2, vy2 = violation_box
        v_center = ((vx1+vx2)//2, (vy1+vy2)//2)
        
        best_rider = None
        min_dist = 99999
        
        for r in rider_data:
            rx1, ry1, rx2, ry2 = r['box']
            # Logic: Violation must be roughly above or inside the rider's box
            if (rx1 < v_center[0] < rx2) and (ry1 - 100 < v_center[1] < ry2):
                dist = abs(v_center[1] - ry1) 
                if dist < min_dist:
                    min_dist = dist
                    best_rider = r # Return the whole rider dict (with ID)
        
        return best_rider

    def detect_violations_full_frame(self, frame, frame_count):
        # --- KEY FIX: Use .track() instead of just detection ---
        # persist=True ensures IDs (1, 2, 3...) stay the same across frames
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False, tracker="botsort.yaml")[0]
        names = results.names
        annotated = frame.copy()
        
        detections = []
        
        # 1. Parse Data (Boxes + IDs)
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().tolist()
            clss = results.boxes.cls.int().cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()
            
            for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                label = names[cls]
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'label': label, 
                    'box': (x1, y1, x2, y2), 
                    'conf': conf, 
                    'id': track_id # We now have the ID!
                })
        elif results.boxes is not None:
             # Fallback if tracker fails momentarily (prevents crash)
             for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                label = names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                detections.append({'label': label, 'box': (x1, y1, x2, y2), 'conf': float(conf), 'id': -1})

        # 2. Filter Lists
        riders = [d for d in detections if d['label'] == 'P_Bike' and d['conf'] >= self.rider_conf]
        no_helmets = [d for d in detections if d['label'] == 'No_Helmet' and d['conf'] >= self.helmet_conf]
        mobiles = [d for d in detections if d['label'] == 'Mobile' and d['conf'] >= self.mobile_conf]
        plates = [d for d in detections if d['label'] == 'LP']

        active_violations = [] # Only for NEW violations to return to stats

        # 3. Draw Riders
        for r in riders:
            x1, y1, x2, y2 = r['box']
            # Draw cyan box for riders
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 4. Match Violations to Riders
        
        # --- CHECK HELMETS ---
        for nh in no_helmets:
            matched_rider = self.assign_violation_to_rider(nh['box'], riders)
            
            if matched_rider:
                # Draw RED box (Visual feedback is always good)
                x1, y1, x2, y2 = nh['box']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f"No Helmet {nh['conf']:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # --- DEDUPLICATION LOGIC ---
                r_id = matched_rider['id']
                
                # Only count if we haven't seen this Rider ID violate before
                # AND ensure the ID is valid (not -1)
                if r_id != -1 and r_id not in self.processed_violation_ids:
                    self.processed_violation_ids.add(r_id) # Mark as caught
                    
                    active_violations.append({
                        'violation': 'No Helmet',
                        'box': matched_rider['box']
                    })

        # --- CHECK MOBILES ---
        for mob in mobiles:
            matched_rider = self.assign_violation_to_rider(mob['box'], riders)
            
            if matched_rider:
                x1, y1, x2, y2 = mob['box']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(annotated, f"Mobile {mob['conf']:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                r_id = matched_rider['id']
                
                if r_id != -1 and r_id not in self.processed_violation_ids:
                    self.processed_violation_ids.add(r_id)
                    active_violations.append({
                        'violation': 'Mobile Usage',
                        'box': matched_rider['box']
                    })

        # 5. Extract Evidence for NEW violations only
        current_frame_evidence = []
        for viol in active_violations:
            rx1, ry1, rx2, ry2 = viol['box']
            
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, rx1-20), max(0, ry1-20)
            cx2, cy2 = min(w, rx2+20), min(h, ry2+20)
            rider_crop = frame[cy1:cy2, cx1:cx2]
            
            # OCR Check
            plate_text = "Unknown"
            for p in plates:
                px1, py1, px2, py2 = p['box']
                # Check if plate is inside/near rider box
                if px1 > rx1 and px2 < rx2 and py1 > ry1 and py2 < ry2:
                    plate_crop = frame[py1:py2, px1:px2]
                    try:
                        ocr_res = self.reader.readtext(plate_crop)
                        text = " ".join([res[1] for res in ocr_res])
                        clean = ''.join(e for e in text if e.isalnum())
                        if len(clean) > 4: plate_text = clean.upper()
                    except: pass
            
            # Save File
            count_val = len(self.processed_violation_ids)
            screenshot_path = f"saved_violations/viol_{frame_count}_{count_val}.jpg"
            cv2.imwrite(screenshot_path, rider_crop)
            
            current_frame_evidence.append({
                'image': rider_crop,
                'plate': plate_text,
                'violations': [viol['violation']]
            })
            
        return annotated, current_frame_evidence