import cv2
import numpy as np
import copy

class ROIDrawer:
    ROI_TYPE_LINE = 0
    ROI_TYPE_POLYGON = 1

    def __init__(self, window_name="Video", distance_threshold=10):
        """
        Initialize flexible ROI Drawer
        
        :param window_name: Name of the window to draw ROIs on
        :param distance_threshold: Pixel distance to close a polygon
        """
        self.window_name = window_name
        self.distance_threshold = distance_threshold
        
        # Drawing state variables
        self.drawing = False
        self.current_roi_points = []
        self.current_roi_type = None
        
        # Store ROIs
        self.rois = {
            'lines': [],
            'polygons': []
        }
        
        # Attach mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for drawing ROIs
        
        :param event: OpenCV mouse event
        :param x: x-coordinate of mouse
        :param y: y-coordinate of mouse
        :param param: Additional parameters (can include current frame)
        """
        frame = param[0] if param else None
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_left_click(x, y, frame)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if frame is not None:
                self._update_drawing_preview(x, y, frame)

    def _handle_left_click(self, x, y, frame=None):
        """
        Handle left mouse button click for ROI drawing
        
        :param x: x-coordinate of click
        :param y: y-coordinate of click
        :param frame: Current frame for preview
        """
        if not self.drawing:
            # Start drawing
            self.drawing = True
            self.current_roi_points = [(x, y)]
            
            # Prompt user to specify ROI type if not already set
            if self.current_roi_type is None:
                print("Press 'l' for line, 'p' for polygon")
        else:
            # Continue drawing based on ROI type
            if self.current_roi_type == self.ROI_TYPE_LINE:
                # For lines, second click ends drawing
                if len(self.current_roi_points) == 1:
                    self.current_roi_points.append((x, y))
                    self.rois['lines'].append(self.current_roi_points)
                    self.drawing = False
                    self.current_roi_type = None
            
            elif self.current_roi_type == self.ROI_TYPE_POLYGON:
                # For polygons, check if close to start point
                if len(self.current_roi_points) > 1 and self._is_close_to_start(x, y):
                    # Close the polygon
                    self.current_roi_points.append(self.current_roi_points[0])
                    self.rois['polygons'].append(self.current_roi_points)
                    self.drawing = False
                    self.current_roi_type = None
                else:
                    # Continue adding points
                    self.current_roi_points.append((x, y))

    def _is_close_to_start(self, x, y):
        """
        Check if the current point is close to the starting point
        
        :param x: x-coordinate of current point
        :param y: y-coordinate of current point
        :return: Boolean indicating if point is close to start
        """
        start_point = self.current_roi_points[0]
        return np.linalg.norm(np.array(start_point) - np.array((x, y))) <= self.distance_threshold

    def _update_drawing_preview(self, x, y, frame):
        """
        Update the preview of the ROI being drawn
        
        :param x: current x-coordinate
        :param y: current y-coordinate
        :param frame: Frame to draw on
        """
        temp_image = frame.copy()
        if self.current_roi_points:
            preview_points = self.current_roi_points + [(x, y)]
            
            if self.current_roi_type == self.ROI_TYPE_LINE:
                # Line preview
                cv2.line(temp_image, preview_points[0], preview_points[1], (0, 255, 255), 2)
            elif self.current_roi_type == self.ROI_TYPE_POLYGON:
                # Polygon preview
                cv2.polylines(temp_image, [np.array(preview_points)], 
                              False, (0, 255, 255), 2)
        
        cv2.imshow(self.window_name, temp_image)

    def set_roi_type(self, roi_type):
        """
        Set the type of ROI to draw
        
        :param roi_type: ROI type (LINE or POLYGON)
        """
        self.current_roi_type = roi_type

    def draw_rois_on_frame(self, frame):
        """
        Draw all ROIs on the given frame
        
        :param frame: Frame to draw ROIs on
        :return: Frame with ROIs drawn
        """
        # Draw lines
        for idx, line in enumerate(self.rois['lines'], 1):
            cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
            cv2.putText(frame, f'Line {idx}', line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw polygons
        for idx, roi in enumerate(self.rois['polygons'], 1):
            cv2.polylines(frame, [np.array(roi)], True, (255, 0, 0), 2)
            # Calculate centroid for ROI label
            centroid = np.mean(roi[:-1], axis=0).astype(int)
            cv2.putText(frame, f'ROI {idx}', tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame

    def delete_last_roi(self):
        """
        Delete the last drawn ROI (line or polygon)
        """
        if self.rois['polygons']:
            self.rois['polygons'].pop()
        elif self.rois['lines']:
            self.rois['lines'].pop()

    def delete_all_rois(self):
        """
        Delete all ROIs
        """
        self.rois['lines'].clear()
        self.rois['polygons'].clear()

    def get_rois(self):
        """
        Get all ROIs
        
        :return: Dictionary of ROIs
        """
        return self.rois

    def get_yolo_counter_rois(self, resized_frame_size, original_frame_size):
        """
        Prepare ROIs for YOLO object counter, scaled to the original frame size.
        
        :param resized_frame_size: Tuple (width, height) of the resized frame.
        :param original_frame_size: Tuple (width, height) of the original frame.
        :return: Dictionary with scaled line and polygon ROIs.
        """
        # Compute scaling factors
        scale_x = original_frame_size[0] / resized_frame_size[0]
        scale_y = original_frame_size[1] / resized_frame_size[1]

        scaled_rois = {'lines': [], 'polygons': []}

        # Scale line ROIs
        for line in self.rois['lines']:
            scaled_line = [
                (int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in line
            ]
            scaled_rois['lines'].append(scaled_line)

        # Scale polygon ROIs
        for polygon in self.rois['polygons']:
            scaled_polygon = [
                (int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in polygon
            ]
            scaled_rois['polygons'].append(scaled_polygon)

        return scaled_rois


    @staticmethod
    def is_point_in_polygon(point, polygon):
        """
        Check if a point is inside a polygon
        
        :param point: Point to check
        :param polygon: Polygon vertices
        :return: Boolean indicating if point is in polygon
        """
        return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0

    @staticmethod
    def is_crossing_line(start, end, line):
        """
        Check if a line segment crosses another line
        
        :param start: Start point of moving object
        :param end: End point of moving object
        :param line: Reference line
        :return: Boolean indicating if line is crossed
        """
        p1, p2 = line
        return (
            (p1[1] - p2[1]) * (start[0] - p1[0]) + (p2[0] - p1[0]) * (start[1] - p1[1]) > 0
        ) != (
            (p1[1] - p2[1]) * (end[0] - p1[0]) + (p2[0] - p1[0]) * (end[1] - p1[1]) > 0
        )
    
    @staticmethod
    def print_ROIs(rois):
        """
        Print all ROIs with their coordinates and types
        """
        print("Current ROIs:")
        
        # Print lines
        if rois['lines']:
            print("\nLines:")
            for idx, line in enumerate(rois['lines'], 1):
                print(f"  Line {idx}: {line[0]}, {line[1]}")

        # Print polygons
        if rois['polygons']:
            print("\nPolygons:")
            for idx, polygon in enumerate(rois['polygons'], 1):
                coords_str = ', '.join(f"({x}, {y})" for x, y in polygon)
                print(f"  Polygon {idx}: {coords_str}")
        else:
            print("  No ROIs defined yet.")


def main():
    # Video capture setup
    cap = cv2.VideoCapture("video/day1.mp4")
    assert cap.isOpened(), "Error reading video file"

    # Create ROI Drawer
    roi_drawer = ROIDrawer()

    while True:
        ret, fr = cap.read()
        if not ret:
            break

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        # Resize frame
        frame = cv2.resize(fr, (w // 3, h // 3))        
        
        # Draw existing ROIs
        frame = roi_drawer.draw_rois_on_frame(frame)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # ROI drawing and deletion controls
        if key == ord('l'):  # Start drawing line
            roi_drawer.set_roi_type(ROIDrawer.ROI_TYPE_LINE)
        elif key == ord('p'):  # Start drawing polygon
            roi_drawer.set_roi_type(ROIDrawer.ROI_TYPE_POLYGON)
        elif key == ord('d'):  # Delete last ROI
            roi_drawer.delete_last_roi()
        elif key == ord('r'):  # Delete all ROIs
            roi_drawer.delete_all_rois()
        elif key == 27:  # ESC key to cancel current drawing
            roi_drawer.drawing = False
            roi_drawer.current_roi_points = []
            roi_drawer.current_roi_type = None
        elif key == ord('o'):
            roi_drawer.get_yolo_counter_rois((w//3, h//3),(w,h))
            roi_drawer.print_ROIs(roi_drawer.get_yolo_counter_rois((w//3, h//3),(w,h)))
        elif key == ord('q'):  # Quit
            break

        # Pass frame to mouse callback to enable drawing preview
        cv2.setMouseCallback("Video", roi_drawer._mouse_callback, [frame])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()