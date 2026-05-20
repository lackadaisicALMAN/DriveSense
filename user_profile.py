"""
user_profile.py  –  DriveSense AI  |  User Vehicle Profile Management
=======================================================================
Manages user vehicle preferences:
  • Store multiple vehicles associated with user
  • Track usage frequency (primary vehicle detection)
  • Persist preferences to local JSON storage
  • Seamless multi-vehicle support with smart defaults
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path


# Local storage path for user profiles
PROFILES_DIR = Path(os.path.expanduser("~/.drivesense"))
USER_PROFILE_FILE = PROFILES_DIR / "user_vehicles.json"


def ensure_storage():
    """Ensure the storage directory exists."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


class UserProfile:
    """Manages a user's vehicle collection and preferences."""

    def __init__(self):
        """Initialize user profile, loading from disk if it exists."""
        ensure_storage()
        self.vehicles: Dict[str, dict] = {}  # vehicle_name -> {braking_time, usage_count}
        self.primary_vehicle: Optional[str] = None
        self._load()

    def _load(self):
        """Load profile from JSON file."""
        if USER_PROFILE_FILE.exists():
            try:
                data = json.load(open(USER_PROFILE_FILE, "r"))
                self.vehicles = data.get("vehicles", {})
                self.primary_vehicle = data.get("primary_vehicle")
            except (json.JSONDecodeError, IOError):
                self.vehicles = {}
                self.primary_vehicle = None

    def _save(self):
        """Persist profile to JSON file."""
        ensure_storage()
        data = {
            "vehicles": self.vehicles,
            "primary_vehicle": self.primary_vehicle,
        }
        with open(USER_PROFILE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def add_vehicle(self, vehicle_name: str, braking_time_s: float):
        """
        Add or update a vehicle in the user's collection.
        
        Args:
            vehicle_name: Name of the vehicle (from VEHICLE_DATABASE)
            braking_time_s: User's measured 0-100 km/h braking time
        """
        self.vehicles[vehicle_name] = {
            "braking_time": braking_time_s,
            "usage_count": self.vehicles.get(vehicle_name, {}).get("usage_count", 0),
        }
        # If this is the first vehicle, set it as primary
        if self.primary_vehicle is None:
            self.primary_vehicle = vehicle_name
        self._save()

    def remove_vehicle(self, vehicle_name: str):
        """Remove a vehicle from the collection."""
        if vehicle_name in self.vehicles:
            del self.vehicles[vehicle_name]
        if self.primary_vehicle == vehicle_name:
            # Switch to the next most-used vehicle
            self.primary_vehicle = self.get_most_used_vehicle()
        self._save()

    def record_usage(self, vehicle_name: str):
        """Record that a vehicle was used for analysis."""
        if vehicle_name in self.vehicles:
            self.vehicles[vehicle_name]["usage_count"] += 1
            self._save()

    def get_most_used_vehicle(self) -> Optional[str]:
        """Return the most frequently used vehicle."""
        if not self.vehicles:
            return None
        return max(self.vehicles.keys(), key=lambda v: self.vehicles[v].get("usage_count", 0))

    def get_vehicle_list(self) -> List[str]:
        """Return list of stored vehicles."""
        return sorted(list(self.vehicles.keys()))

    def get_suggested_vehicle(self) -> Optional[str]:
        """
        Return the suggested vehicle for next upload.
        Strategy:
          1. If only one vehicle: suggest it
          2. If multiple: suggest the most-used one
          3. Otherwise: return primary vehicle
        """
        vehicle_list = self.get_vehicle_list()
        if len(vehicle_list) == 1:
            return vehicle_list[0]
        
        most_used = self.get_most_used_vehicle()
        if most_used and most_used in self.vehicles:
            return most_used
        
        return self.primary_vehicle

    def get_vehicle_braking_time(self, vehicle_name: str) -> float:
        """Get the recorded braking time for a vehicle."""
        if vehicle_name in self.vehicles:
            return self.vehicles[vehicle_name].get("braking_time", 3.5)
        return 3.5  # fallback default
