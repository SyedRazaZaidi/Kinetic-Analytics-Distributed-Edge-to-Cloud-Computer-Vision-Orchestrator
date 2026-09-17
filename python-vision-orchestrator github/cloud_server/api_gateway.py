import asyncio
import logging
import json
import cv2
import numpy as np
import time
import base64
import sqlite3
import csv
from io import StringIO
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import zmq
import zmq.asyncio
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# --- ENTERPRISE DATABASE INITIALIZATION ---
db_conn = sqlite3.connect("kinetic_analytics.db", check_same_thread=False)
cursor = db_conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS agent_logs 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, category TEXT, event TEXT)''')
db_conn.commit()

# --- LOAD DUAL MODELS ---
logger.info("Initializing Dual Vision Engines (Pose + Object Detection)...")
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

INTERACTIVE_OBJECT_CLASSES = [39, 41, 56, 58, 62, 63, 64, 65, 66, 67, 73]

zmq_context = zmq.asyncio.Context()
active_websockets = []
latest_payload = ""

# --- AGENT STATE MANAGEMENT ---
incident_log = []
last_logged_state = {"action": "", "interaction": "", "time": 0}

def log_event(category, event_text):
    """Logs autonomous events to SQLite and the live in-memory terminal."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO agent_logs (timestamp, category, event) VALUES (?, ?, ?)", 
                   (current_time, category, event_text))
    db_conn.commit()
    
    timestamp_short = datetime.now().strftime("%H:%M:%S")
    incident_log.insert(0, f"[{timestamp_short}] {event_text}")
    if len(incident_log) > 15:
        incident_log.pop()

def run_ai_inference(frame):
    """Executes dual inference and heuristic math, returning the annotated frame AND a telemetry dictionary."""
    global last_logged_state
    height, width = frame.shape[:2]
    current_time = time.time()
    
    pose_results = pose_model.track(frame, persist=True, classes=[0], verbose=False, tracker="botsort.yaml")
    annotated_frame = pose_results[0].plot()

    obj_results = object_model(frame, classes=INTERACTIVE_OBJECT_CLASSES, verbose=False, conf=0.35)
    
    detected_objects = []
    obj_boxes_raw = []
    if obj_results[0].boxes is not None:
        boxes = obj_results[0].boxes.xyxy.cpu().numpy()
        class_ids = obj_results[0].boxes.cls.int().cpu().tolist()
        confs = obj_results[0].boxes.conf.cpu().tolist()
        names = obj_results[0].names

        for box, cls_id, conf in zip(boxes, class_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            obj_name = names[cls_id].upper()
            label = f"{obj_name} {conf:.2f}"
            detected_objects.append(obj_name)
            obj_boxes_raw.append((box, obj_name))
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 191, 0), 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + len(label) * 9, y1), (255, 191, 0), -1)
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # --- 1. HEURISTIC GESTURE ENGINE ---
    current_action = "IDLE"
    if hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
        for kp in pose_results[0].keypoints.xy.cpu().numpy():
            if len(kp) >= 11:
                l_shoulder, r_shoulder = kp[5], kp[6]
                l_wrist, r_wrist = kp[9], kp[10]
                
                if (l_wrist[1] > 0 and l_wrist[1] < l_shoulder[1]) or (r_wrist[1] > 0 and r_wrist[1] < r_shoulder[1]):
                    current_action = "ATTENTION / HAND RAISED"

    # --- 2. CONTEXTUAL INTERACTION TRACKING ---
    current_interaction = "NONE"
    if pose_results[0].boxes is not None and len(pose_results[0].boxes.xyxy) > 0:
        person_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
        for p_box in person_boxes:
            px1, py1, px2, py2 = p_box
            for o_box, o_name in obj_boxes_raw:
                ox1, oy1, ox2, oy2 = o_box
                if ox1 < px2 and ox2 > px1 and oy1 < py2 and oy2 > py1:
                    current_interaction = f"USING {o_name}"

    # --- 3. AUTOMATED DIAGNOSTIC AGENT ---
    if current_action != "IDLE" and (current_time - last_logged_state["time"] > 3 or last_logged_state["action"] != current_action):
        log_event("GESTURE", f"Subject performing: {current_action}")
        last_logged_state["action"] = current_action
        last_logged_state["time"] = current_time

    if current_interaction != "NONE" and (current_time - last_logged_state["time"] > 3 or last_logged_state["interaction"] != current_interaction):
        log_event("BEHAVIOR", f"Subject engaged with: {current_interaction}")
        last_logged_state["interaction"] = current_interaction
        last_logged_state["time"] = current_time

    # Center target reticle
    center_x, center_y = width // 2, height // 2
    cv2.line(annotated_frame, (center_x - 18, center_y), (center_x + 18, center_y), (255, 255, 255), 1)
    cv2.line(annotated_frame, (center_x, center_y - 18), (center_x, center_y + 18), (255, 255, 255), 1)
    cv2.circle(annotated_frame, (center_x, center_y), 32, (56, 189, 248), 1)

    persons_detected = len(pose_results[0].boxes.id) if (pose_results[0].boxes is not None and pose_results[0].boxes.id is not None) else 0

    # Package clean telemetry data
    telemetry_data = {
        "persons": persons_detected,
        "objects": len(detected_objects),
        "heuristic": current_action,
        "synergy": current_interaction
    }

    return annotated_frame, telemetry_data

async def video_stream_listener():
    global latest_payload
    socket = zmq_context.socket(zmq.SUB)
    socket.bind("tcp://0.0.0.0:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    logger.info("ZeroMQ Video Listener bound to tcp://0.0.0.0:5555")
    
    try:
        while True:
            multipart_data = await socket.recv_multipart()
            while await socket.poll(timeout=0):
                multipart_data = await socket.recv_multipart()
            
            if len(multipart_data) == 2:
                metadata_bytes, frame_bytes = multipart_data
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue
                
                annotated_frame, telemetry = await asyncio.to_thread(run_ai_inference, frame)
                
                _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                payload = {
                    "image": img_base64,
                    "telemetry": telemetry,
                    "logs": incident_log[:10]
                }
                latest_payload = json.dumps(payload)
                
                for ws in active_websockets.copy():
                    try: await ws.send_text(latest_payload)
                    except Exception: active_websockets.remove(ws)
    except asyncio.CancelledError:
        logger.info("ZeroMQ Listener shutting down gracefully.")
    finally:
        socket.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    listener_task = asyncio.create_task(video_stream_listener())
    yield
    listener_task.cancel()
    zmq_context.term()
    db_conn.close()

app = FastAPI(title="Kinetic Analytics Orchestrator", lifespan=lifespan)

def generate_html_page(active_tab, title, main_content, script_content=""):
    nav_home = 'active' if active_tab == 'home' else ''
    nav_live = 'active' if active_tab == 'live' else ''
    nav_image = 'active' if active_tab == 'image' else ''
    nav_video = 'active' if active_tab == 'video' else ''
    nav_data = 'active' if active_tab == 'data' else ''

    return f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>KIA | {title}</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --bg-base: #030712;
                    --bg-panel: rgba(17, 24, 39, 0.7);
                    --border: rgba(255, 255, 255, 0.1);
                    --accent: #38bdf8;
                    --accent-glow: rgba(56, 189, 248, 0.4);
                    --text-main: #f8fafc;
                    --text-muted: #94a3b8;
                    --success: #10b981;
                }}
                
                body {{
                    font-family: 'Inter', sans-serif;
                    background: radial-gradient(circle at top left, #1e293b, var(--bg-base));
                    color: var(--text-main);
                    margin: 0; padding: 24px; display: flex; height: 100vh; box-sizing: border-box; overflow: hidden;
                }}
                
                .sidebar {{
                    width: 280px; background: var(--bg-panel); backdrop-filter: blur(16px);
                    -webkit-backdrop-filter: blur(16px); border: 1px solid var(--border);
                    border-radius: 16px; padding: 24px; display: flex; flex-direction: column;
                    gap: 24px; margin-right: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                }}
                
                .logo-container h1 {{
                    font-size: 22px; font-weight: 700; margin: 0 0 8px 0;
                    background: linear-gradient(90deg, #38bdf8, #818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 0.5px;
                }}
                
                .status-indicator {{
                    display: inline-block; width: 8px; height: 8px; background: var(--success);
                    border-radius: 50%; margin-right: 8px; box-shadow: 0 0 10px var(--success);
                }}
                
                .nav-menu {{ display: flex; flex-direction: column; gap: 8px; }}
                .nav-item {{
                    text-decoration: none; color: var(--text-muted); padding: 12px 16px;
                    border-radius: 8px; font-weight: 600; font-size: 14px; transition: all 0.3s ease;
                    border: 1px solid transparent; display: flex; align-items: center; gap: 12px;
                }}
                .nav-item:hover {{ background: rgba(255,255,255,0.05); color: var(--text-main); }}
                .nav-item.active {{
                    background: rgba(56, 189, 248, 0.1); color: var(--accent);
                    border: 1px solid rgba(56, 189, 248, 0.3); box-shadow: 0 0 15px var(--accent-glow);
                }}
                
                .main-content {{ flex: 1; display: flex; flex-direction: column; overflow-y: auto; padding-right: 10px; }}
                
                .video-container {{
                    flex: 1; background-color: #000; border-radius: 16px; overflow: hidden;
                    border: 1px solid var(--border); box-shadow: 0 20px 40px rgba(0,0,0,0.6);
                    display: flex; align-items: center; justify-content: center;
                }}
                .video-container img {{ width: 100%; height: 100%; object-fit: contain; }}
                
                .upload-box {{
                    flex: 1; background: var(--bg-panel); backdrop-filter: blur(16px);
                    border-radius: 16px; border: 2px dashed rgba(56, 189, 248, 0.3); display: flex;
                    flex-direction: column; align-items: center; justify-content: center;
                    color: var(--text-muted); transition: all 0.3s ease;
                }}
                .upload-box:hover {{ border-color: var(--accent); background: rgba(56, 189, 248, 0.05); }}
                
                .home-header {{ margin-bottom: 30px; }}
                .home-title {{ font-size: 36px; font-weight: 700; margin: 0 0 12px 0; }}
                .home-subtitle {{ font-size: 16px; color: var(--text-muted); max-width: 700px; line-height: 1.6; margin: 0; }}
                
                .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }}
                .feature-card {{
                    background: var(--bg-panel); backdrop-filter: blur(16px);
                    border: 1px solid var(--border); border-radius: 16px; padding: 24px;
                    transition: transform 0.3s ease, border-color 0.3s ease;
                }}
                .feature-card:hover {{ transform: translateY(-5px); border-color: rgba(56, 189, 248, 0.4); }}
                .feature-icon {{
                    width: 48px; height: 48px; background: rgba(56, 189, 248, 0.1);
                    border-radius: 12px; display: flex; align-items: center; justify-content: center;
                    color: var(--accent); margin-bottom: 20px;
                }}

                .telemetry-sidebar {{
                    width: 320px; background: var(--bg-panel); backdrop-filter: blur(16px);
                    border: 1px solid var(--border); border-radius: 16px; padding: 24px;
                    display: flex; flex-direction: column; gap: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                }}
                .metric-box {{
                    display: flex; justify-content: space-between; align-items: center;
                    padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.05);
                }}
            </style>
        </head>
        <body>
            <div class="sidebar">
                <div class="logo-container">
                    <h1>KINETIC ANALYTICS</h1>
                    <div style="font-size: 13px; color: var(--text-muted); font-weight: 600;">
                        <span class="status-indicator"></span>SYSTEM ONLINE
                    </div>
                </div>
                
                <div class="nav-menu">
                    <a href="/" class="nav-item {nav_home}">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
                        Platform Overview
                    </a>
                    <a href="/live-telemetry" class="nav-item {nav_live}">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                        Live Edge Stream
                    </a>
                    <a href="/image-analysis" class="nav-item {nav_image}">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                        Static Image Analysis
                    </a>
                    <a href="/video-analysis" class="nav-item {nav_video}">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                        Batch Video Processing
                    </a>
                    <a href="/database-logs" class="nav-item {nav_data}">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path></svg>
                        Time-Series Logs
                    </a>
                </div>
                
                <div class="metric-card" style="margin-top: 15px; background: rgba(0,0,0,0.2); padding: 14px; border-radius: 8px; border-left: 4px solid var(--accent);">
                    <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Vision Engines</div>
                    <div style="font-size: 13px; font-weight: bold; color: var(--text-main);">Synergy (Math + Box)</div>
                </div>
                
                <div style="margin-top: auto; padding-top: 20px; border-top: 1px solid var(--border);">
                    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Infrastructure Target</div>
                    <div style="font-weight: 600; font-size: 14px; color: var(--text-main);">tcp://0.0.0.0:5555</div>
                </div>
            </div>

            <div class="main-content">
                {main_content}
            </div>

            <script>
                {script_content}
            </script>
        </body>
    </html>
    """

@app.get("/")
async def get_home_page():
    main_content = """
        <div class="home-header">
            <h1 class="home-title">Enterprise Vision Analytics</h1>
            <p class="home-subtitle">An end-to-end distributed computer vision orchestrator. Kinetic Analytics leverages edge video streaming, real-time 17-point pose estimation, spatial bounding-box synergy, and autonomous event logging.</p>
        </div>
        
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon"><svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg></div>
                <h3 style="margin: 0 0 10px 0; font-size: 18px;">Heuristic Gesture Math</h3>
                <p style="margin: 0; font-size: 14px; color: var(--text-muted); line-height: 1.6;">Translates raw 17-point skeletal coordinates into contextual actions (e.g., raised hands) using dynamic Y-axis keypoint thresholding.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon"><svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg></div>
                <h3 style="margin: 0 0 10px 0; font-size: 18px;">Contextual Object Synergy</h3>
                <p style="margin: 0; font-size: 14px; color: var(--text-muted); line-height: 1.6;">Calculates continuous intersection over union (IoU) overlaps between human and object boundaries to detect complex interactions.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon"><svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4"></path></svg></div>
                <h3 style="margin: 0 0 10px 0; font-size: 18px;">Automated SQLite Diagnostic</h3>
                <p style="margin: 0; font-size: 14px; color: var(--text-muted); line-height: 1.6;">A background intelligence agent logs unique behaviors into a time-series database while rendering a live terminal feed to the user.</p>
            </div>
        </div>
    """
    return HTMLResponse(generate_html_page('home', 'Overview', main_content))

@app.get("/live-telemetry")
async def get_live_dashboard():
    main_content = """
        <div style="display: flex; flex-direction: row; gap: 24px; height: 100%; width: 100%;">
            <div class="video-container" style="flex: 1;">
                <img id="live-frame" alt="Awaiting ZeroMQ Telemetry from WebCam..." />
            </div>
            
            <div class="telemetry-sidebar">
                <h2 style="margin: 0; font-size: 18px; color: var(--text-main); border-bottom: 1px solid var(--border); padding-bottom: 15px;">LIVE TELEMETRY</h2>
                
                <div class="metric-box">
                    <span style="font-size: 13px; color: var(--text-muted);">Human Targets</span>
                    <span id="telemetry-persons" style="font-size: 16px; font-weight: bold; color: var(--accent);">0</span>
                </div>
                <div class="metric-box">
                    <span style="font-size: 13px; color: var(--text-muted);">Objects Active</span>
                    <span id="telemetry-objects" style="font-size: 16px; font-weight: bold; color: var(--success);">0</span>
                </div>
                <div class="metric-box">
                    <span style="font-size: 13px; color: var(--text-muted);">Heuristic Action</span>
                    <span id="telemetry-heuristic" style="font-size: 14px; font-weight: bold; color: var(--text-main);">IDLE</span>
                </div>
                <div class="metric-box">
                    <span style="font-size: 13px; color: var(--text-muted);">Contextual Synergy</span>
                    <span id="telemetry-synergy" style="font-size: 14px; font-weight: bold; color: var(--text-main);">NONE</span>
                </div>
                
                <h3 style="margin: 10px 0 0 0; font-size: 14px; color: #818cf8; border-bottom: 1px solid var(--border); padding-bottom: 10px;">DIAGNOSTIC TERMINAL</h3>
                <div id="terminal-logs" style="flex: 1; overflow-y: auto; font-family: monospace; font-size: 11px; display: flex; flex-direction: column; gap: 8px;">
                    <!-- Agent logs injected here via WebSocket JSON -->
                </div>
            </div>
        </div>
    """
    script_content = """
        var ws = new WebSocket("ws://localhost:8000/ws/stream");
        var imageElement = document.getElementById('live-frame');
        
        ws.onmessage = function(event) {
            var data = JSON.parse(event.data);
            
            imageElement.src = "data:image/jpeg;base64," + data.image;
            
            document.getElementById('telemetry-persons').innerText = data.telemetry.persons;
            document.getElementById('telemetry-objects').innerText = data.telemetry.objects;
            
            var heuristicEl = document.getElementById('telemetry-heuristic');
            heuristicEl.innerText = data.telemetry.heuristic;
            heuristicEl.style.color = data.telemetry.heuristic !== "IDLE" ? "var(--success)" : "var(--text-main)";
            
            var synergyEl = document.getElementById('telemetry-synergy');
            synergyEl.innerText = data.telemetry.synergy;
            synergyEl.style.color = data.telemetry.synergy !== "NONE" ? "var(--accent)" : "var(--text-main)";
            
            var terminalEl = document.getElementById('terminal-logs');
            terminalEl.innerHTML = "";
            data.logs.forEach(function(log) {
                var color = "var(--text-main)";
                if(log.includes("GESTURE")) color = "var(--success)";
                if(log.includes("BEHAVIOR")) color = "var(--accent)";
                terminalEl.innerHTML += `<div style="color: ${color}; line-height: 1.4;">${log}</div>`;
            });
        };
    """
    return HTMLResponse(generate_html_page('live', 'Live Telemetry', main_content, script_content))

@app.get("/image-analysis")
async def get_image_page():
    main_content = """
        <div class="upload-box" id="drop-zone">
            <svg style="width: 64px; height: 64px; margin-bottom: 20px; color: var(--accent);" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
            <h2 id="upload-title" style="margin: 0 0 8px 0; color: var(--text-main);">Upload Analysis Image</h2>
            <p id="upload-desc" style="margin: 0 0 20px 0;">Select an image to run a full dual-engine synergy analysis.</p>
            <input type="file" id="imageInput" accept="image/*">
        </div>
        <div id="result-container" style="display: none; flex: 1; flex-direction: column; height: 100%;">
            <div style="background: var(--bg-panel); backdrop-filter: blur(16px); padding: 16px 24px; border-radius: 16px 16px 0 0; border: 1px solid var(--border); border-bottom: none; display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: var(--text-main);">Analysis Complete</h3>
                <span id="detection-count" style="background: rgba(16, 185, 129, 0.2); color: var(--success); padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px; border: 1px solid rgba(16, 185, 129, 0.4);">Found: 0</span>
            </div>
            <div class="video-container" style="border-radius: 0 0 16px 16px; border-top: none; height: 100%;">
                <img id="result-image" style="width: 100%; height: 100%; object-fit: contain;" />
            </div>
            <button onclick="location.reload()" style="margin-top: 20px; padding: 14px; background: var(--accent); color: #000; border: none; border-radius: 8px; font-weight: 700; font-size: 15px; cursor: pointer;">Analyze Another Image</button>
        </div>
    """
    script_content = """
        document.getElementById('imageInput').addEventListener('change', async function(e) {
            if(e.target.files.length === 0) return;
            const formData = new FormData();
            formData.append('file', e.target.files[0]);
            
            document.getElementById('upload-title').innerText = "Analyzing Image...";
            document.getElementById('upload-desc').innerText = "Running dual YOLO models on Cloud Brain.";
            document.getElementById('imageInput').style.display = 'none';
            
            try {
                const response = await fetch('/api/analyze-image', { method: 'POST', body: formData });
                const data = await response.json();
                
                document.getElementById('drop-zone').style.display = 'none';
                document.getElementById('result-container').style.display = 'flex';
                document.getElementById('result-image').src = data.annotated_image;
                document.getElementById('detection-count').innerText = `Analysis: Complete`;
            } catch(err) {
                alert("Analysis failed. Ensure server is running.");
                location.reload();
            }
        });
    """
    return HTMLResponse(generate_html_page('image', 'Image Analysis', main_content, script_content))

@app.get("/video-analysis")
async def get_video_page():
    main_content = """
        <div class="upload-box" id="drop-zone">
            <svg style="width: 64px; height: 64px; margin-bottom: 20px; color: var(--accent);" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"></path></svg>
            <h2 id="upload-title" style="margin: 0 0 8px 0; color: var(--text-main);">Batch Video Processing</h2>
            <p id="upload-desc" style="margin: 0 0 20px 0;">Upload an MP4 file for frame-by-frame autonomous synergy tracking.</p>
            <input type="file" id="videoInput" accept="video/mp4">
        </div>
        <div id="result-container" style="display: none; flex: 1; flex-direction: column; height: 100%;">
            <div style="background: var(--bg-panel); backdrop-filter: blur(16px); padding: 16px 24px; border-radius: 16px 16px 0 0; border: 1px solid var(--border); border-bottom: none; display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: var(--text-main);">Video Inference Stream</h3>
                <span style="background: rgba(56, 189, 248, 0.2); color: var(--accent); padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px; border: 1px solid rgba(56, 189, 248, 0.4);">PROCESSING</span>
            </div>
            <div class="video-container" style="border-radius: 0 0 16px 16px; border-top: none; height: 100%;">
                <img id="video-stream" style="width: 100%; height: 100%; object-fit: contain;" />
            </div>
            <button onclick="location.reload()" style="margin-top: 20px; padding: 14px; background: var(--accent); color: #000; border: none; border-radius: 8px; font-weight: 700; font-size: 15px; cursor: pointer;">Analyze Another Video</button>
        </div>
    """
    script_content = """
        document.getElementById('videoInput').addEventListener('change', async function(e) {
            if(e.target.files.length === 0) return;
            const formData = new FormData();
            formData.append('file', e.target.files[0]);
            
            document.getElementById('upload-title').innerText = "Uploading Video...";
            document.getElementById('upload-desc').innerText = "Transferring payload to Cloud Brain.";
            document.getElementById('videoInput').style.display = 'none';
            
            try {
                const response = await fetch('/api/upload-video', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.status === "success") {
                    document.getElementById('drop-zone').style.display = 'none';
                    document.getElementById('result-container').style.display = 'flex';
                    document.getElementById('video-stream').src = data.stream_url;
                }
            } catch(err) {
                alert("Upload failed.");
                location.reload();
            }
        });
    """
    return HTMLResponse(generate_html_page('video', 'Video Analysis', main_content, script_content))

@app.get("/database-logs")
async def get_logs_page():
    """Route 5: The SQLite Time-Series Database View"""
    cursor.execute("SELECT timestamp, category, event FROM agent_logs ORDER BY id DESC LIMIT 50")
    logs = cursor.fetchall()
    
    rows_html = ""
    for row in logs:
        color = "var(--success)" if row[1] == "GESTURE" else "var(--accent)" if row[1] == "BEHAVIOR" else "var(--text-main)"
        rows_html += f"""
        <tr style="border-bottom: 1px solid var(--border);">
            <td style="padding: 12px; color: var(--text-muted);">{row[0]}</td>
            <td style="padding: 12px; font-weight: bold; color: {color};">{row[1]}</td>
            <td style="padding: 12px;">{row[2]}</td>
        </tr>
        """
        
    main_content = f"""
        <div style="background: var(--bg-panel); border-radius: 16px; border: 1px solid var(--border); padding: 24px; flex: 1; overflow-y: auto;">
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                <h2 style="margin: 0; color: var(--text-main);">SQLite Time-Series Intelligence</h2>
                <a href="/api/export-analytics" download style="text-decoration: none; background: var(--success); color: #000; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 14px; display: flex; align-items: center; gap: 8px; transition: background 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                    <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    Export CSV for Power BI
                </a>
            </div>
            
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="background: rgba(0,0,0,0.3);">
                        <th style="padding: 12px; color: var(--text-muted);">Timestamp</th>
                        <th style="padding: 12px; color: var(--text-muted);">Entity Category</th>
                        <th style="padding: 12px; color: var(--text-muted);">Autonomous Diagnostic Event</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html if rows_html else '<tr><td colspan="3" style="padding: 20px; text-align: center; color: var(--text-muted);">No events logged yet.</td></tr>'}
                </tbody>
            </table>
        </div>
    """
    return HTMLResponse(generate_html_page('data', 'Database Logs', main_content))


# --- API ENDPOINTS FOR PROCESSING & EXPORT ---

@app.get("/api/export-analytics")
async def export_analytics():
    """Dumps the SQLite time-series logs into a downloadable CSV format for Power BI ingestion."""
    cursor.execute("SELECT id, timestamp, category, event FROM agent_logs ORDER BY id DESC")
    logs = cursor.fetchall()
    
    # Write to an in-memory string buffer using Python's native CSV builder
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Category", "Event"]) # Standard headers
    writer.writerows(logs)
    
    # Return directly as a downloadable file stream
    response = Response(content=output.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=kinetic_analytics_export.csv"
    return response

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    annotated_frame, _ = await asyncio.to_thread(run_ai_inference, frame)
    
    _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "annotated_image": f"data:image/jpeg;base64,{img_base64}",
        "detections": "Complete"
    }

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    file_location = "temp_batch_video.mp4"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"status": "success", "stream_url": "/api/stream-batch-video"}

def generate_batch_frames():
    cap = cv2.VideoCapture("temp_batch_video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, _ = run_ai_inference(frame)
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.get("/api/stream-batch-video")
async def stream_batch_video():
    return StreamingResponse(generate_batch_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    if latest_payload:
        await websocket.send_text(latest_payload)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)