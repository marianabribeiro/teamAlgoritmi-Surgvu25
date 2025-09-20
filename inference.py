

import cv2
import os
import json
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import tarfile
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import re

#######################################################################################################
# EXTRACT TAR.GZ MODEL
#######################################################################################################

MODEL_DIR = "model"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
INPUT_PATH=Path("input")

OUTPUT_PATH=Path("output")

models = [
    YOLO("model_weights/best_deepSort.pt"),   # model 1
    YOLO("model_weights/model72.pt"), # model 2
    YOLO("model_weights/best_ann.pt"),
    YOLO("model_weights/best.pt")
]


# Load YOLO model
#model_path = os.path.join(MODEL_DIR, "runs_AUG/lp/train7/weights/best.pt")
#model=YOLO("model/best_deepSort.pt")
#model=YOLO("model/model72.pt")
#model = YOLO("model/best.pt")
#model = YOLO("model/best_ann.pt")
#######################################################################################################
# OBJECT DETECTION
#######################################################################################################
def detect_objects(frame, models):
    """
    Run detection on the same frame using multiple YOLO models and combine results.
    """
    combined_boxes = []

    for model in models:
        results = model.predict(frame, verbose=False)[0]
        for box in results.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            name = results.names[int(box.cls)]
            prob = float(box.conf)

            combined_boxes.append({
                "corners": [
                    [round(float(x_min), 2), round(float(y_min), 2), 0.5],
                    [round(float(x_max), 2), round(float(y_min), 2), 0.5],
                    [round(float(x_max), 2), round(float(y_max), 2), 0.5],
                    [round(float(x_min), 2), round(float(y_max), 2), 0.5],
                ],
                "name": normalize_name(name),
                "probability": round(prob, 3)
            })

    # --- Remove duplicates (overlapping boxes of the same class) ---
    final_boxes = []
    for box in combined_boxes:
        is_duplicate = False
        for fb in final_boxes:
            if box["name"] == fb["name"] and boxes_are_close(box, fb, threshold=30):
                # Keep the one with higher probability
                if box["probability"] > fb["probability"]:
                    fb.update(box)
                is_duplicate = True
                break
        if not is_duplicate:
            final_boxes.append(box)

    return final_boxes


#######################################################################################################
# HELPER FUNCTIONS
#######################################################################################################
REPEATED_TOOLS = ['needle_driver', 'forceps']

def normalize_name(name: str) -> str:
    """Normalize tool names."""
    return str(name).strip().lower().replace(" ", "_")

def filter_boxes_by_limit(boxes, max_tools=3):
    """Limit the number of tools detected per frame."""
    if len(boxes) <= max_tools:
        return boxes

    tool_names = [b['name'].split('_', 2)[-1] for b in boxes]
    tool_counter = Counter(tool_names)

    if all(count == 1 for count in tool_counter.values()):
        return sorted(boxes, key=lambda x: x['probability'], reverse=True)[:max_tools]

    for rep_tool in [tool for tool, count in tool_counter.items() if count > 1]:
        if rep_tool in REPEATED_TOOLS:
            print(f"Atenção: ferramenta repetida detectada -> {rep_tool}")

    return boxes


#######################################################################################################
# VIDEO TO JSON
#######################################################################################################
import math

def boxes_are_close(box1, box2, threshold=30):
    """
    Check if two boxes are close to each other (Euclidean distance between centers).
    """
    x1_center = (box1["corners"][0][0] + box1["corners"][2][0]) / 2
    y1_center = (box1["corners"][0][1] + box1["corners"][2][1]) / 2
    x2_center = (box2["corners"][0][0] + box2["corners"][2][0]) / 2
    y2_center = (box2["corners"][0][1] + box2["corners"][2][1]) / 2
    distance = math.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)
    return distance <= threshold



def refine_missing_detections(annotations, video_frames, models, start_conf=0.6, min_conf=0.1, step=0.1):
    """
    Refine frames with missing detections based on temporal consistency.
    Now uses multiple models and keeps only the highest-confidence detections.
    """
    frame_keys = sorted(annotations.keys())
    for idx, frame_key in enumerate(frame_keys):
        current_boxes = annotations[frame_key]["boxes"]

        # Skip if already has detections
        if len(current_boxes) > 0:
            continue

        # Get neighbor frames
        prev_boxes = annotations[frame_keys[idx - 1]]["boxes"] if idx > 0 else []
        next_boxes = annotations[frame_keys[idx + 1]]["boxes"] if idx < len(frame_keys) - 1 else []

        # If both neighbors are empty, skip
        if len(prev_boxes) == 0 and len(next_boxes) == 0:
            continue

        #print(f"Refining frame {frame_key}: No detections but neighbors have objects...")

        frame_number = int(frame_key.replace("frame", ""))
        frame = video_frames[frame_number]

        # Start re-inference with decreasing confidence
        conf = start_conf
        while conf >= min_conf:
            combined_boxes = []

            for model in models:
                results = model.predict(frame, conf=conf, verbose=False)[0]
                for det_nr, box in enumerate(results.boxes, start=1):
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    name = results.names[int(box.cls)]
                    prob = float(box.conf)
                    combined_boxes.append({
                        "corners": [
                            [round(float(x_min), 2), round(float(y_min), 2), 0.5],
                            [round(float(x_max), 2), round(float(y_min), 2), 0.5],
                            [round(float(x_max), 2), round(float(y_max), 2), 0.5],
                            [round(float(x_min), 2), round(float(y_max), 2), 0.5],
                        ],
                        "name": f"slice_nr_{det_nr}_{normalize_name(name)}",
                        "probability": round(prob, 3)
                    })

            # Keep only the best (highest confidence) detections
            if len(combined_boxes) > 0:
                best_boxes = sorted(combined_boxes, key=lambda b: b["probability"], reverse=True)
                annotations[frame_key]["boxes"] = best_boxes[:3]  # still limit to top 3
                #print(f"Recovered {len(annotations[frame_key]['boxes'])} detection(s) for {frame_key} at conf={conf}")
                break

            conf -= step

    return annotations




def video_to_json(video_path, output_json, min_confidence=0.1, pad=3):
    cap = cv2.VideoCapture(video_path)
    annotations = {}
    frames_list = []  # store frame keys for adjacency checks
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_key = f"frame{frame_idx:0{pad}d}"
        frames_list.append(frame_key)
        detections = detect_objects(frame, models)

        # Build initial box list
        boxes = []
        #print(boxes)
        for det_nr, det in enumerate(detections, start=1):
            if det["probability"] < float(min_confidence):
                continue
            boxes.append({
                "corners": det["corners"],
                "name": f"slice_nr_{det_nr}_{normalize_name(det['name'])}",
                "probability": det["probability"]
            })

        # --- Post-processing per frame ---
        # 1. Limit to max 3 detections (keep top confidences)
        if len(boxes) > 3:
            boxes = sorted(boxes, key=lambda b: b["probability"], reverse=True)[:3]

        # 2. Check for overlapping detections of the same object, keep the most confident
        filtered_boxes = []
        for b in boxes:
            keep = True
            for fb in filtered_boxes:
                if normalize_name(b["name"]) == normalize_name(fb["name"]) and boxes_are_close(b, fb):
                    keep = False
                    break
            if keep:
                filtered_boxes.append(b)
        boxes = filtered_boxes

        # Save annotations for current frame
        annotations[frame_key] = {
            "type": "Multiple 2D bounding boxes",
            "boxes": boxes,
            "version": {"major": 1, "minor": 0}
        }

    cap.release()

    # --- Post-processing across frames ---
    for i, frame_key in enumerate(frames_list):
        current_boxes = annotations[frame_key]["boxes"]

        # Check for frames with 0 detections
        if len(current_boxes) == 0 and i + 1 < len(frames_list):
            next_frame_key = frames_list[i + 1]
            next_boxes = annotations[next_frame_key]["boxes"]
            if len(next_boxes) == 0:
                print(f"Frames {frame_key} and {next_frame_key} both have 0 detections (possible issue).")

    # Save final JSON
    #with open(output_json, "w") as f:
    #    json.dump(annotations, f, indent=4)
   # print(f"JSON output: {output_json}")
    return annotations

def remove_single_frame_detections(annotations, iou_threshold=0.3):
    """
    1. Remove detections that appear in only one frame across the video.
    2. Remove overlapping boxes in the same frame (keep the highest confidence).
       Applies to both same-class and different-class overlaps.
    """
    # Count occurrences of each detection name across all frames
    occurrence_counter = Counter()
    for frame_data in annotations.values():
        for box in frame_data["boxes"]:
            occurrence_counter[normalize_name(box["name"])] += 1

    def iou(box1, box2):
        """Compute Intersection over Union (IoU) between two boxes."""
        x1_min, y1_min = box1["corners"][0][0], box1["corners"][0][1]
        x1_max, y1_max = box1["corners"][2][0], box1["corners"][2][1]

        x2_min, y2_min = box2["corners"][0][0], box2["corners"][0][1]
        x2_max, y2_max = box2["corners"][2][0], box2["corners"][2][1]

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    # Apply filters
    for frame_key, frame_data in annotations.items():
        # Step 1: Remove detections that only appear once in the whole video
        filtered_boxes = [
            box for box in frame_data["boxes"]
            if occurrence_counter[normalize_name(box["name"])] > 1
        ]

        # Step 2: Remove overlapping boxes (same or different classes)
        final_boxes = []
        for box in filtered_boxes:
            keep = True
            for fb in final_boxes:
                if iou(box, fb) > iou_threshold:
                    # keep the one with higher probability
                    if box["probability"] > fb["probability"]:
                        fb.update(box)
                    keep = False
                    break
            if keep:
                final_boxes.append(box)

        annotations[frame_key]["boxes"] = final_boxes

    return annotations


###### convert to right format



FRAME_RE = re.compile(r'frame0*(\d+)', re.IGNORECASE)

def normalize_token(full_name: str) -> str:
    """
    Drop the first 3 '_' parts (slice, nr, frameNr) and join the rest.
    """
    if not full_name:
        return "unknown"
    parts = full_name.split("_")
    return "_".join(parts[3:]) if len(parts) > 3 else parts[-1]

def extract_tools(per_frame_ann: dict) -> list:
    """
    Extrai lista única de ferramentas detetadas no vídeo a partir das anotações por frame.
    """
    ferramentas = set()
    for frame_val in per_frame_ann.values():
        if not isinstance(frame_val, dict):
            continue
        boxes = frame_val.get("boxes", [])
        for box in boxes:
            orig_name = box.get("name") or ""
            ferramentas.add(normalize_token(orig_name))
    return sorted(list(ferramentas))


#######################################################################################################
# MAIN LOOP
#######################################################################################################
from pathlib import Path


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # Load inputs
    # -----------------------------
    #question_file =  "/input/visual-context-question.json" 
    import os
    question_file = os.path.join("/input", "visual-context-question.json")
    #print("question file!! --> ", question_file)
    with open(question_file, "r", encoding="utf-8") as f:
        question = json.load(f)

    #print(f"Loaded question: {question}")


    
    

    # -----------------------------
    # Process the video
    # -----------------------------
    input_path = "/input/endoscopic-robotic-surgery-video.mp4"

    # Read frames for refinement
    video_frames = {}
    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        video_frames[frame_idx] = frame.copy()
    cap.release()

    # Run detection pipeline
    annotations = video_to_json(input_path, None, min_confidence=0.6, pad=3)

    annotations = refine_missing_detections(
        annotations,
        video_frames,
        models,
        start_conf=0.6,
        min_conf=0.1,
        step=0.1
    )
    annotations = remove_single_frame_detections(annotations)

    # Extract detected tools
    ferramentas_y = extract_tools(annotations)
    #print(f"Ferramentas detetadas: {ferramentas_detectadas}")
    #tools = ["bipolar_forceps", "monopular_curved_scissors"]
    ferramentas_detectadas = [tool.replace("_", " ") for tool in ferramentas_y]

   # print(cleaned_tools)


    # -----------------------------
    # Combine detections with the question
    # -----------------------------
    
    #print(f"Question: {question}")

    # Example: build context for QA pipeline
    tools = f"In this video, these tools are used: {', '.join(ferramentas_detectadas)}."
    doc_texts = [
        "Procedure: Endoscopic surgery involves small incisions where robotic instruments and a camera are inserted. The surgeon controls the instruments from a console."
        "The summary is describing endoscopic or laparoscopic surgery.",
        "An endoscopic surgery is not an open surgery.",
        "location of the procedure: inside the body.",
        "Only three robotic instruments are usually present in the images during surgery.",
        "Monopolar curved scissors are used for cutting, blunt and sharp dissection, electrocautery, and suturing.",
        "Permanent monopolar cautery hook is used for precise dissection and division of tissue with monopolar cautery.",
        "Vessel sealer is a disposable instrument designed for sealing and cutting vessels and tissue bundles.",
        "Force bipolar is used for dissection, grasping, retraction, and bipolar coagulation of tissue.",
        "The forceps are used for grasping and holding tissues or objects.",
        "Suction irrigator delivers fluid to the surgical site and evacuates fluids; it can also be used for retraction and blunt dissection.",
        "Clip applier is used for applying polymer locking clips for vessel and tissue ligation.",
        "Needle driver is used for suturing and knot tying in endoscopic surgery.",
        "If a needle driver is used, suturing is probably necessary.",
        "Task: Suturing involves the process of passing a needle through tissue and tying knots with robotic tools.",
        "Task: Uterine horn dissection uses monopolar curved scissors for cut and coagulation.",
        "Task: Suspensory ligaments dissection involves blunt dissection at different abdominal depths.",
        "Task: Rectal artery/vein isolation requires careful dissection to separate vessels from organ structures.",
        "Task: Range of motion involves navigating the endoscope and tools to avoid collisions inside the cavity."
    ]


    examples = """
        Example 1
        Context: Monopolar curved scissors are used for cutting, blunt and sharp dissection, electrocautery, and suturing.
        Question: "Is tissue being cut during this clip?"
        Answer: "Yes, the clip shows tissue being cut."

        Example 2
        Context: Laparoscopy procedures are performed on the Uterine horn or bladder.
        Question: What organ is being manipulated in the clip?
        Answer: The organ being manipulated is the uterine horn.

        Example 3
        Context: Needle driver is used for suturing and knot tying in endoscopic surgery.
        Question: What is the purpose of the needle driver?
        Answer: It is used to pass the needle through tissue and tie knots.

        Example 4
        Context: A Needle driver is used for suturing and knot tying in endoscopic surgery. There is a needle driver in the clip.
        Question: Was suturing necessary? Was suturing a part of the process?
        Answer: Yes, a suture is necessary.

        Example 5
        Context: Endoscopic surgery involves small incisions where robotic instruments and a camera are inserted.
        Question: What procedure is the video showing?
        Answer: This appears to be endoscopic or laparoscopic surgery.

        Example 6
        Question: "Is there a needle driver in the clip?"
        Answer: "Yes, there is a needle driver."
        """


    # Criar embeddings
    embedder = SentenceTransformer('models/all-MiniLM-L6-v2')
    embeddings = embedder.encode(doc_texts)
    
    # Criar índice FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # ---------------------------
    # 4. Prompt for QA
    # ---------------------------
    
    # Criamos uma query (objetos + pergunta)
    query_text = f"{tools}. Question: {question}"
    query_embedding = embedder.encode([query_text])
    
    # Recuperamos os 2 chunks mais relevantes
    query_embedding = embedder.encode([query_text])
    D, I = index.search(query_embedding, k=5)  # retrieve top 5 chunks
    contexto_relevante = " ".join([doc_texts[i] for i in I[0]])
        

    qa_pipeline = pipeline("text2text-generation", model="models/flan-t5-base")

    input_text = (
        "You are a medical assistant specialized in robotic surgery. "
        "Answer the question using ONLY the context provided. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"{examples}\n\n"
        f"Context:\n{contexto_relevante}\n\n"
        f"Question: {query_text}\nAnswer:"
    )

    resposta = qa_pipeline(input_text, max_length=128, do_sample=False)
    output = resposta[0]['generated_text']


    response_path = os.path.join("/output", "visual-context-response.json")


    print("Answer:", resposta[0]['generated_text'])
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    #print(f"Response saved in: {response_path}")








