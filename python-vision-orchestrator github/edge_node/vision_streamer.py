import cv2
import zmq
import time
import json
import logging
from multiprocessing import Process, Queue, Event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EDGE-%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

def capture_worker(frame_queue: Queue, stop_event: Event, source: int = 0):
    """Continuously pulls frames from the sensor into an in-memory queue."""
    # Point back to the live physical webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("CRITICAL: Cannot open webcam. Check hardware connections!")
        stop_event.set()
        return

    # Enforce standard acquisition parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger.info("Video capture worker started reading from WEBCAM (Live Telemetry)")
    sequence_id = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Dropped frame from webcam hardware.")
            time.sleep(0.01)
            continue

        sequence_id += 1
        payload = {
            "sequence_id": sequence_id,
            "timestamp": time.time(),
            "frame": frame
        }

        # Keep queue fresh: if full, drop oldest frame to prevent latency drift
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Exception:
                pass
        frame_queue.put(payload)

    cap.release()
    logger.info("Capture worker terminated.")

def streamer_worker(frame_queue: Queue, stop_event: Event, target_host: str = "tcp://127.0.0.1:5555"):
    """Encodes frames to JPEG and dispatches multipart packets via ZeroMQ."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # Set High-Water Mark to 2: drop packets immediately if network backs up
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.connect(target_host)
    logger.info(f"Connected publisher to {target_host}")

    # Standard JPEG compression parameters
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.005)
            continue

        data = frame_queue.get()
        frame = data["frame"]

        # Compress to binary buffer
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            continue

        metadata = json.dumps({
            "camera_id": "edge_cam_01",
            "sequence_id": data["sequence_id"],
            "timestamp": data["timestamp"],
            "byte_size": buffer.nbytes
        }).encode("utf-8")

        # Multipart message: Part 1 = Metadata JSON, Part 2 = JPEG binary
        socket.send_multipart([metadata, buffer.tobytes()])

    socket.close()
    context.term()
    logger.info("Streamer worker terminated.")

if __name__ == "__main__":
    stop_event = Event()
    # Buffer capacity locked to 2 to eliminate frame buffering delays
    frame_pipeline = Queue(maxsize=2)

    p_capture = Process(target=capture_worker, args=(frame_pipeline, stop_event, 1))
    p_stream = Process(target=streamer_worker, args=(frame_pipeline, stop_event, "tcp://127.0.0.1:5555"))

    p_capture.start()
    p_stream.start()

    logger.info("Kinetic Intersection Pipeline running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cleaning up processes...")
        stop_event.set()
        p_capture.join(timeout=2)
        p_stream.join(timeout=2)
        logger.info("Edge node clean exit completed.")