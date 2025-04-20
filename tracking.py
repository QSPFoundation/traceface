# Copyright (C) 2025 Val Argunov (byte AT qsp DOT org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import asyncio
import time
import math
import json
import cv2
import numpy as np
import mediapipe as mp
import socket
import signal
import logging

class FaceTracker:
    def __init__(self, args):
        self.args = args
        self.camera = self.init_camera()
        self.stop = asyncio.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.listener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listener_sock.setblocking(False)
        try:
            self.listener_sock.bind(('0.0.0.0', self.args.port))
            logging.info(f"Successfully bound to UDP port {self.args.port}")
        except Exception as e:
            logging.error(f"Failed to bind UDP port {self.args.port}: {e}")
            self.stop.set()

        self.target_ip = None
        self.target_ports = []

        # See https://github.com/DenchiSoft/VTubeStudioBlendshapeUDPReceiverTest/tree/main
        self.cur_data = {
            "FaceFound": False,
            "Position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "Rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "BlendShapes": []
        }

        self.detector = self.init_face_detector()
        self.fps_delay = 1/self.args.fps

    def close(self):
        self.camera.release()
        self.sock.close()
        self.listener_sock.close()

    def init_camera(self):
        logging.info(f"Initializing camera {self.args.camera}")
        camera = cv2.VideoCapture(
            self.args.camera,
            cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
        )
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera

    def init_face_detector(self):
        logging.info(f"Initializing face detector model ({self.args.model})")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            num_faces = 1,
            base_options = mp.tasks.BaseOptions(model_asset_path = self.args.model),
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback = self.process_face_result
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process_face_result(self, result: mp.tasks.vision.FaceLandmarkerResult, *args):
        try:
            self.update_blendshapes(result)
            self.update_head_position(result)
            self.cur_data["FaceFound"] = True
        except:
            logging.warning("Face not detected")
            self.cur_data["FaceFound"] = False

    def get_bs_score(self, bs, index):
        if self.args.smoothing > 0:
            current_value = self.cur_data["BlendShapes"][index]["v"] if self.cur_data["BlendShapes"] else 0
            return current_value * self.args.smoothing + bs.score * (1 - self.args.smoothing)
        else:
            return bs.score

    def update_blendshapes(self, result):
        blendshapes = result.face_blendshapes[0]
        self.cur_data["BlendShapes"] = [{"k": bs.category_name, "v": self.get_bs_score(bs, index)} for index, bs in enumerate(blendshapes)]

    def update_head_position(self, result):
        mat = np.array(result.facial_transformation_matrixes[0])

        to_degrees = 180 / math.pi
        pitch = np.arctan2(-mat[2, 0], np.hypot(mat[2, 1], mat[2, 2])) * to_degrees
        yaw = np.arctan2(mat[1, 0], mat[0, 0]) * to_degrees
        roll = np.arctan2(mat[2, 1], mat[2, 2]) * to_degrees

        self.cur_data["Position"].update(x = -mat[0][3], y = mat[1][3], z = mat[2][3])
        self.cur_data["Rotation"].update(x = pitch, z = yaw, y = roll)

    async def send_facial_data(self):
        logging.info("Starting data sending loop")
        try:
            while not self.stop.is_set():
                if self.target_ip and self.target_ports:
                    start = time.time()
                    data_json = json.dumps(self.cur_data).encode()
                    for port in self.target_ports:
                        self.sock.sendto(data_json, (self.target_ip, port))
                    await asyncio.sleep(max(0, self.fps_delay - (time.time() - start)))
                else:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in send_facial_data: {e}")
            self.stop.set()
        finally:
            logging.info("Stopped data sending loop")

    async def capture_frames(self):
        logging.info("Starting frame capture loop")
        try:
            while not self.stop.is_set():
                start = time.time()
                ret, frame = await asyncio.get_event_loop().run_in_executor(None, self.camera.read)
                if not ret:
                    logging.error("Camera error - failed to read frame")
                    self.stop.set()
                    break

                # We have to send images in RGB format
                image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)

                # Use live detection to keep up with processing
                self.detector.detect_async(image, int(start * 1000))

                await asyncio.sleep(max(0, self.fps_delay - (time.time() - start)))
        except Exception as e:
            logging.error(f"Error in capture_frames: {e}")
            self.stop.set()
        finally:
            logging.info("Stopped frame capture loop")

    async def listen_for_requests(self):
        logging.info("Starting UDP listener")
        loop = asyncio.get_event_loop()
        try:
            while not self.stop.is_set():
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(self.listener_sock, 1024),
                        timeout=0.5
                    )
                    try:
                        message = json.loads(data.decode())
                        if message.get("messageType") == "iOSTrackingDataRequest":
                            required_keys = ["ports"]
                            if all(key in message for key in required_keys):
                                ports = list(map(int, message["ports"]))
                                self.target_ip = addr[0]
                                self.target_ports = ports
                                logging.info(f"Configured target: {self.target_ip} ports {self.target_ports}")
                            else:
                                logging.warning("Received malformed request - missing required keys")
                    except Exception as e:
                        logging.error(f"Message processing error: {e}")
                except asyncio.TimeoutError:
                    # Timeout is normal - just check if we need to exit
                    continue
                except Exception as e:
                    logging.error(f"UDP listener error: {e}")
                    self.stop.set()
        finally:
            logging.info("Stopped UDP listener")

    async def run(self):
        logging.info("Starting main tracking loop")
        try:
            tasks = [
                self.capture_frames(),
                self.send_facial_data(),
                self.listen_for_requests()
            ]
            await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            raise
        finally:
            logging.info("Main tracking loop stopped")

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} must be between 0.0 and 1.0")
    return x

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="Face tracker")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Index of the camera device")
    parser.add_argument("-p", "--port", type=int, default=21412, help="Port to listen")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--model", default="face_landmarker.task", help="Face model to use")
    parser.add_argument("--smoothing", type=restricted_float, default=0.0, help="Smoothing percentage (0.0-1.0)")
    args = parser.parse_args()

    logging.info(f"Starting tracker with config: Camera={args.camera} Port={args.port} FPS={args.fps}")

    tracker = FaceTracker(args)
    loop = asyncio.new_event_loop()

    try:
        loop.add_signal_handler(signal.SIGINT, tracker.stop.set)
        loop.add_signal_handler(signal.SIGTERM, tracker.stop.set)
    except NotImplementedError:
        logging.warning("Signal handlers are not supported on this platform")

    try:
        loop.run_until_complete(tracker.run())
    except Exception as e:
        logging.error(f"Runtime error: {e}")
    finally:
        logging.info("Cleaning up resources")
        tracker.close()
        loop.close()
        logging.info("Shutdown completed")

if __name__ == "__main__":
    main()