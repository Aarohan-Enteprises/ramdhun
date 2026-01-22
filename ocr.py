import cv2
import numpy as np
import matplotlib. pyplot as plt

def detect_face_circle_accurate(image_path):
    """Detect the specific circular face photo in the poster"""
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    
    print(f"Image dimensions: {width} x {height}")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Focus on the lower portion of the image where the face circle is
    # Create a mask for the bottom half
    mask = np.zeros_like(gray)
    mask[int(height * 0.6):, :] = 255  # Only look at bottom 40% of image
    
    # Apply mask
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=mask)
    
    # Detect circles with more restrictive parameters
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=200,  # Increased to avoid multiple detections
        param1=100,    # Higher threshold for edge detection
        param2=30,     # Higher threshold for circle detection
        minRadius=70,  # Minimum radius for face circle
        maxRadius=110  # Maximum radius for face circle
    )
    
    print("\n=== Circle Detection Results ===")
    
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        
        # Filter circles that are in the bottom-right region
        valid_circles = []
        for circle in circles[0, :]: 
            x, y, r = circle[0], circle[1], circle[2]
            # Check if circle is in the bottom-right quadrant area
            if y > height * 0.7 and x > width * 0.4:
                valid_circles.append((x, y, r))
        
        if valid_circles:
            print(f"Found {len(valid_circles)} face circle(s) in target region")
            
            for i, (x, y, r) in enumerate(valid_circles):
                print(f"\n✓ Face Circle {i+1}:")
                print(f"  Center: ({x}, {y})")
                print(f"  Radius: {r}")
                print(f"  Bounding Box: Top-left=({x-r}, {y-r}), Bottom-right=({x+r}, {y+r})")
                
                # Draw the detected circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 3)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 5)
                
                # Add coordinate label
                cv2.putText(output, f"Center: ({x}, {y})", 
                           (x - 80, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(output, f"Radius: {r}", 
                           (x - 50, y - r - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(output, (x-r, y-r), (x+r, y+r), (255, 0, 0), 2)
        else:
            print("No circles found in the target region (bottom-right area)")
            print("\nTrying face detection as fallback...")
            
            # Fallback:  Face detection in bottom portion
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            bottom_region = img[int(height * 0.6):, :]
            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(80, 80)
            )
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Adjust coordinates to full image
                    y_adjusted = y + int(height * 0.6)
                    center_x = x + w//2
                    center_y = y_adjusted + h//2
                    radius = max(w, h) // 2
                    
                    print(f"\n✓ Face detected:")
                    print(f"  Center: ({center_x}, {center_y})")
                    print(f"  Approximate radius: {radius}")
                    
                    cv2.circle(output, (center_x, center_y), radius, (0, 255, 0), 3)
                    cv2.circle(output, (center_x, center_y), 2, (0, 0, 255), 5)
    else:
        print("No circles detected")
    
    # Display results
    plt.figure(figsize=(16, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Poster')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Detected Face Circle')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('corrected_detection. png', dpi=150, bbox_inches='tight')
    print("\n✓ Result saved as 'corrected_detection.png'")
    plt.show()
    
    cv2.imwrite('output_corrected.jpg', output)
    print("✓ Output saved as 'output_corrected.jpg'")

if __name__ == "__main__":
    image_path = "poster.jpg"
    
    print("Detecting circular face photo in poster...")
    print("="*50)
    
    detect_face_circle_accurate(image_path)
    
    print("\n" + "="*50)
    print("Detection complete!")