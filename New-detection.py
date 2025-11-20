from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

# Initialize detection network (SSD-MobileNet-v2, threshold = 0.5 for confidence)
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("/dev/video0")  # V4L2 camera (adjust if using USB camera)
display = videoOutput("display://0") # Live display (replace with "my_video.mp4" to save video)

# Counter to limit output to 2 detection results (adjust as needed)
detection_count = 0

while display.IsStreaming() and detection_count < 2:
    # Capture frame from camera
    img = camera.Capture()
    
    if img is None:  # Skip if frame capture timed out
        continue

    # Run object detection on the frame
    detections = net.Detect(img)

    # Process each detected object (stop after 2 valid detections)
    for det in detections:
        if detection_count >= 2:
            break  # Only extract 2 results

        class_id = det.ClassID          
        confidence = det.Confidence      
        left = det.Left                  
        top = det.Top                    
        right = det.Right               
        bottom = det.Bottom              
        width = right - left             
        height = bottom - top            
        area = width * height            
        center_x = (left + right) / 2  
        center_y = (top + bottom) / 2    

        print(f"\n=== Detection Result {detection_count + 1} ===")
        print(f"ClassID:       {class_id}")
        print(f"Confidence:    {confidence:.2f}")  
        print(f"Left:          {left:.0f}")       
        print(f"Top:           {top:.0f}")
        print(f"Right:         {right:.0f}")
        print(f"Bottom:        {bottom:.0f}")
        print(f"Width:         {width:.0f}")
        print(f"Height:        {height:.0f}")
        print(f"Area:          {area:.0f}")
        print(f"Center (X,Y):  ({center_x:.1f}, {center_y:.1f})")

        detection_count += 1  # Increment counter after valid detection

    # Render frame with bounding boxes and FPS
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

print("\nExtracted 2 detection results. Exiting...")
