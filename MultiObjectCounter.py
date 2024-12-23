import cv2
import numpy as np
import os
from ultralytics import YOLO
from ROIDrawer import ROIDrawer
from ObjectCounter import ObjectCounter
import json
from ultralytics.utils.plotting import Annotator, colors
import csv

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MultiROIObjectCounter:
    def __init__(self, video_path, model_path='models/yolo11n.pt', output_video = 'multi_roi_object_counting_output', vehicle_classes=None):
        """
        Initialize Multi-ROI Object Counter
        
        :param video_path: Path to the input video
        :param model_path: Path to YOLO model
        :param vehicle_classes: List of vehicle class indices to track
        """
        
        # Video capture setup
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), "Error reading video file"
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.annotator = None
        self.roi_path = None
        # Save video method
        self.video_writer = cv2.VideoWriter(
            "multi_roi_object_counting_output.mp4", 
            cv2.VideoWriter_fourcc(*"mp4v"),  # MP4 codec
            self.fps, 
            (self.width, self.height)
        )
        
    # Vehicle classes
        #self.vehicle_classes = vehicle_classes or [1, 2, 3, 5, 6, 7]

        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.names = self.model.names

        self.vehicle_classes = vehicle_classes if vehicle_classes is not None else list(self.names.keys())
        # Initialize ROI Drawer
        self.roi_drawer = ROIDrawer()
        
        # Store original ROI points
        self.original_roi_points = {}

    def setup_rois(self):
        """
        Interactive ROI setup using resized video frames
        """
        print("ROI Setup Instructions:")
        print("- Press 'l' to draw a line ROI")
        print("- Press 'p' to draw a polygon ROI")
        print("- Press 'd' to delete last ROI")
        print("- Press 'r' to reset all ROIs")
        print("- Press 'q' to finish ROI setup")

        while True:
            ret, fr = self.cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(fr, (self.width // 3, self.height // 3))        
            
            # Draw existing ROIs
            frame = self.roi_drawer.draw_rois_on_frame(frame)

            cv2.imshow("Video", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # ROI drawing and deletion controls
            if key == ord('l'):  # Start drawing line
                self.roi_drawer.set_roi_type(ROIDrawer.ROI_TYPE_LINE)
            elif key == ord('p'):  # Start drawing polygon
                self.roi_drawer.set_roi_type(ROIDrawer.ROI_TYPE_POLYGON)
            elif key == ord('d'):  # Delete last ROI
                self.roi_drawer.delete_last_roi()
            elif key == ord('r'):  # Delete all ROIs
                self.roi_drawer.delete_all_rois()
            elif key == 27:  # ESC key to cancel current drawing
                self.roi_drawer.drawing = False
                self.roi_drawer.current_roi_points = []
                self.roi_drawer.current_roi_type = None
            elif key == ord('o'):
                self.roi_drawer.get_yolo_counter_rois((self.width//3, self.height//3),(self.width,self.height))
                self.roi_drawer.print_ROIs(self.roi_drawer.get_yolo_counter_rois((self.width//3, self.height//3),(self.width,self.height)))
            elif key == ord('q'):  # Quit
                break

            # Pass frame to mouse callback to enable drawing preview
            cv2.setMouseCallback("Video", self.roi_drawer._mouse_callback, [frame])

        cv2.destroyAllWindows()
    
        # Store original ROI points
        self.original_roi_points = self.roi_drawer.get_yolo_counter_rois((self.width//3, self.height//3),(self.width,self.height))
        
        self.roi_path = self.save_json_file(self.original_roi_points)
        # Reset video capture to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def save_json_file(self, data, output='./', filename='rois.json'):
        """
        Save data to a JSON file and return the file path.
        
        :param data: Dictionary or list to save as JSON.
        :param output: Output directory (default is current directory).
        :param filename: Name of the JSON file (default is 'rois.json').
        :return: Full file path of the saved JSON file.
        """
        # Ensure the output directory exists
        os.makedirs(output, exist_ok=True)

        # Construct full file path
        filepath = os.path.join(output.rstrip('/'), filename)

        # Save the JSON file
        with open(filepath, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        return filepath

    def read_json_file(self, filepath):
        """
        Read data from a JSON file.
        
        :param filepath: Path to the JSON file.
        :return: Parsed JSON data (as a dictionary or list).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
        
        return data

    # Count Object function        
    def count_objects(self):
        """
        Count objects in multiple ROIs (Regions of Interest).
        """
        # Use original ROI points for full-resolution frame
        results = self.read_json_file(self.roi_path)
        lines = results['lines']
        polygons = results['polygons']

        # Initialize counters for each ROI
        roi_counters = [
            # For lines, initialize ObjectCounter for each line
            {
                "type": "line",
                "index": idx,
                "region": line,
                "counter": ObjectCounter(
                    show=False, 
                    region=line, 
                    classes=self.vehicle_classes, 
                    names=self.model.names
                ),
                "count_in": 0,
                "count_out": 0,
            }
            for idx, line in enumerate(lines)
        ] + [
            # For polygons, initialize ObjectCounter for each polygon
            {
                "type": "polygon",
                "index": idx,
                "region": polygon,
                "counter": ObjectCounter(
                    show=False, 
                    region=polygon, 
                    classes=self.vehicle_classes, 
                    names=self.model.names
                ),
                "count_in": 0,
                "count_out": 0,
            }
            for idx, polygon in enumerate(polygons)
        ]

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            # Perform object tracking
            results = self.model.track(
                source=frame, 
                persist=True, 
                conf=0.5, 
                verbose=False, 
                classes=self.vehicle_classes
            )

            for roi in roi_counters:
                # Count objects within the region
                roi["counter"].count(frame, results)

                # Accumulate counts for IN and OUT
                count_in, count_out = 0, 0
                for class_name, counts in roi["counter"].classwise_counts.items():
                    count_in += counts.get("IN", 0)
                    count_out += counts.get("OUT", 0)

                roi["count_in"] = count_in
                roi["count_out"] = count_out

            line_count = 0
            polygon_count = 0

            # Print count details for each line and polygon ROI
            for idx, roi in enumerate(roi_counters):
                if roi["type"] == "line":
                    # Print the index, line, and count details
                    print(f"Line {line_count + 1} (ROI {idx}):")
                    print(f"  - Count IN: {roi['count_in']}")
                    print(f"  - Count OUT: {roi['count_out']}")
                    print("-" * 30)
                    line_count += 1
                elif roi["type"] == "polygon":
                    # Print the index, polygon, and count details
                    print(f"Polygon {polygon_count + 1} (ROI {idx}):")
                    print(f"  - Count IN: {roi['count_in']}")
                    print(f"  - Count OUT: {roi['count_out']}")
                    print("-" * 30)
                    polygon_count += 1
            

            frame_resized = cv2.resize(frame, (1000, 600))  # Resize for display
            cv2.imshow("Hehehee", frame_resized)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                line_count = 0
                polygon_count = 0
                print("Exiting...")

                # Open a CSV file for writing
                with open("roi_counts.csv", mode="w", newline="") as file:
                    writer = csv.writer(file)
                    # Write the header row
                    writer.writerow(["ROI Type", "ROI Index", "Count IN", "Count OUT"])

                    # Print and save count details for each line and polygon ROI
                    for idx, roi in enumerate(roi_counters):
                        if roi["type"] == "line":
                            # Print the index, line, and count details
                            print(f"Line {line_count + 1} (ROI {idx}):")
                            print(f"  - Count IN: {roi['count_in']}")
                            print(f"  - Count OUT: {roi['count_out']}")
                            print("-" * 30)
                            
                            # Write to CSV
                            writer.writerow(["Line", line_count + 1, roi["count_in"], roi["count_out"]])
                            line_count += 1

                        elif roi["type"] == "polygon":
                            # Print the index, polygon, and count details
                            print(f"Polygon {polygon_count + 1} (ROI {idx}):")
                            print(f"  - Count IN: {roi['count_in']}")
                            print(f"  - Count OUT: {roi['count_out']}")
                            print("-" * 30)
                            
                            # Write to CSV
                            writer.writerow(["Polygon", polygon_count + 1, roi["count_in"], roi["count_out"]])
                            polygon_count += 1

                print("Counts saved to 'roi_counts.csv'.")
                break

            # Optionally, write processed frame to output video (if needed)
            self.video_writer.write(frame_resized)

def main():
    # Create Multi-ROI Object Counter
    counter = MultiROIObjectCounter("video/shop_mall.mp4")
    # Interactive ROI setup
    counter.setup_rois()
    counter.count_objects()
if __name__ == "__main__":
    main()