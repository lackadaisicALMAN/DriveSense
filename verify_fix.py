from detector import DriverProfile, FrameEvent
from scorer import SafetyScorer

profile = DriverProfile(vehicle_name='Test', braking_100_sec=3.5, speed_limit_kmh=70)
scorer = SafetyScorer(profile)
events = [FrameEvent(frame_idx=0, timestamp_s=0.0)]
report = scorer.score(events)
print(report.vehicle_name, report.overall_score)
