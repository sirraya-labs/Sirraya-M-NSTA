from sirraya_api import create_sirraya_api

# Initialize system
system = create_sirraya_api()

# Analyze single frame
result = system.analyze_frame(frame)
sis_score = result['result']['sirraya_integrity_score']
sis_category = result['result']['sis_category']

# Analyze video
video_report = system.analyze_video('path/to/video.mp4')
avg_sis = video_report['aggregated_sis']['video_sirraya_integrity_score']