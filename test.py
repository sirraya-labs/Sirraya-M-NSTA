# test_working.py
from sirraya_api import create_sirraya_api
import numpy as np

# Initialize with force face detection
system = create_sirraya_api()
system.config['force_face_detection'] = True

# Create frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Analyze
result = system.analyze_frame(frame)

# Print simple output
print(f"SIS Score: {result['result']['sirraya_integrity_score']}")
print(f"Verdict: {result['result']['sis_verdict']}")
print(f"Category: {result['result']['sis_category']}")

# Print if face was detected
if 'error' in result['metadata']:
    print(f"Error: {result['metadata']['error']}")
else:
    print("✅ Face detected successfully")