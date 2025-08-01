import cv2
import numpy as np
from record3d import Record3DStream
from threading import Event

# ---- USER-ADJUSTABLE PARAMETERS ----
ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL  # dictionary your marker was generated from
MARKER_ID = 0                             # ID of the printed marker
MARKER_SIZE_M = 0.04                      # 4 cm physical size of the marker
# -------------------------------------
# (No fixed target point: user clicks to query coordinates)
# -------------------------------------

class RGBDArucoApp:
    """Streams RGB-D frames, detects an ArUco marker and converts pixel clicks to the marker frame."""
    def __init__(self):
        self.session = None
        self.event = Event()
        self.depth = None  # latest depth frame for pixel queries
        self.rgb = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # GUI
        self.rgb_window = "RGB"
        self.depth_window = "Depth"
        cv2.namedWindow(self.rgb_window)
        cv2.namedWindow(self.depth_window)
        cv2.setMouseCallback(self.rgb_window, self.on_mouse)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Add the same parameters as the original working script
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.1
        params.maxMarkerPerimeterRate = 0.8
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.1
        params.minDistanceToBorder = 3
        params.minOtsuStdDev = 5.0
        params.perspectiveRemovePixelPerCell = 4
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.maxErroneousBitsInBorderRate = 0.2
        params.minMarkerDistanceRate = 0.05
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.current_pose_T_cam_marker = None  # 4x4 homogeneous transformation
        
        # Storage for last clicked pixel and its marker-frame coordinates
        self.clicked_pixel = None           # (u,v)
        self.clicked_marker = None          # (x,y,z) in marker frame

    # ------------------------------------------------------------------
    # Record3D callbacks
    # ------------------------------------------------------------------
    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    # ------------------------------------------------------------------
    # GUI mouse callback – click on RGB window to convert pixel → marker
    # ------------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks: report pixel's 3-D position in marker frame."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pixel = None
            self.clicked_marker = None
            if self.depth is None or self.current_pose_T_cam_marker is None or self.camera_matrix is None:
                print("⚠️  Cannot compute – depth or marker pose not available")
                return
            # Depth and RGB may have different resolutions – translate RGB click to depth pixel
            h_d, w_d = self.depth.shape[:2]
            h_r, w_r = self.rgb.shape[:2]
            depth_x = int(x * w_d / w_r)
            depth_y = int(y * h_d / h_r)
            if not (0 <= depth_y < h_d and 0 <= depth_x < w_d):
                print("⚠️  Click outside frame")
                return
            depth_val = float(self.depth[depth_y, depth_x])
            if depth_val <= 0:
                print("⚠️  Invalid depth at pixel")
                return

            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            X = (x - cx) * depth_val / fx
            Y = (y - cy) * depth_val / fy
            Z = depth_val
            pt_cam = np.array([X, Y, Z, 1.0])
            pt_marker = np.linalg.inv(self.current_pose_T_cam_marker) @ pt_cam
            self.clicked_pixel = (x, y)
            self.clicked_marker = pt_marker[:3]
            print(f"RGB pixel ({x},{y}) → Depth pixel ({depth_x},{depth_y}) depth {depth_val:.3f} m → Camera ({X:.3f},{Y:.3f},{Z:.3f}) → Marker ({pt_marker[0]:.3f},{pt_marker[1]:.3f},{pt_marker[2]:.3f}) m")

    # ------------------------------------------------------------------
    def connect_to_device(self, dev_idx: int = 0):
        print("Searching for devices …")
        devs = Record3DStream.get_connected_devices()
        if not devs:
            raise RuntimeError("No Record3D devices found!")
        if dev_idx >= len(devs):
            raise RuntimeError(f"Device index {dev_idx} out of range")
        dev = devs[dev_idx]
        print(f"Connecting to {dev.udid} (id {dev.product_id}) …")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)

    # ------------------------------------------------------------------
    def get_camera_matrix(self):
        coeffs = self.session.get_intrinsic_mat()
        return np.array([[coeffs.fx, 0, coeffs.tx],
                         [0, coeffs.fy, coeffs.ty],
                         [0, 0, 1]], dtype=np.float32)

    # ------------------------------------------------------------------
    def estimate_pose_from_corners(self, corners_2d):
        # marker corner positions in marker coordinate system (origin at top-left)
        obj_pts = np.array([[0, 0, 0],
                            [MARKER_SIZE_M, 0, 0],
                            [MARKER_SIZE_M, MARKER_SIZE_M, 0],
                            [0, MARKER_SIZE_M, 0]], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, corners_2d, self.camera_matrix, self.dist_coeffs)
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    # ------------------------------------------------------------------
    def run(self):
        while True:
            self.event.wait()
            # fetch frames
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            if depth is None or rgb is None:
                self.event.clear()
                continue
            # store for callbacks
            self.depth = depth
            self.rgb = rgb
            if self.camera_matrix is None:
                self.camera_matrix = self.get_camera_matrix()
                print("Camera intrinsics loaded:", self.camera_matrix)

            # detect marker
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is not None and MARKER_ID in ids.flatten():
                idx = list(ids.flatten()).index(MARKER_ID)
                self.current_pose_T_cam_marker = self.estimate_pose_from_corners(corners[idx][0])
                # draw marker and its axis if pose valid
                if self.current_pose_T_cam_marker is not None:
                    cv2.aruco.drawDetectedMarkers(rgb, [corners[idx]])
                    cv2.drawFrameAxes(rgb, self.camera_matrix, self.dist_coeffs, cv2.Rodrigues(self.current_pose_T_cam_marker[:3,:3])[0], self.current_pose_T_cam_marker[:3,3], MARKER_SIZE_M*0.5)
                    # If a pixel was clicked, draw its marker coordinate near the pixel
                    if self.clicked_pixel and self.clicked_marker is not None:
                        click_u, click_v = self.clicked_pixel
                        cv2.circle(rgb, (click_u, click_v), 6, (0, 0, 255), 2)
                        label = f"({self.clicked_marker[0]:.2f},{self.clicked_marker[1]:.2f},{self.clicked_marker[2]:.2f})m"
                        cv2.putText(rgb, label, (click_u + 8, click_v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            else:
                self.current_pose_T_cam_marker = None

            # display frames
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.rgb_window, rgb_bgr)
            cv2.imshow(self.depth_window, depth)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # No numeric shortcuts needed in this mode
            self.event.clear()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = RGBDArucoApp()
    app.connect_to_device(dev_idx=0)
    app.run() 