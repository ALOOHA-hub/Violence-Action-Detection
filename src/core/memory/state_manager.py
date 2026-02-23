import time

class ThreatState:
    def __init__(self):
        self.level = 0           # 0: Green (Safe), 1: Orange (Suspect), 2: Red (Confirmed)
        self.label = "analyzing" # Text to display on the bounding box
        self.vlm_summary = None  # The detailed JSON report from Qwen
        self.last_seen = time.time()
        self.strike_count = 0    # How many times Phase 2 triggered

class SecurityStateManager:
    def __init__(self, alert_trigger_count=3):
        self.states = {} # Maps tracker_id to ThreatState
        self.trigger_count = alert_trigger_count

    def _ensure_exists(self, tracker_id):
        if tracker_id not in self.states:
            self.states[tracker_id] = ThreatState()
        self.states[tracker_id].last_seen = time.time()
        return self.states[tracker_id]

    def update_phase2(self, tracker_id, is_violent, action_label, confidence):
        """Called multiple times per second by the fast CoCa model."""
        state = self._ensure_exists(tracker_id)
        
        # INFINITE LATCH: If already Red, completely ignore Phase 2 updates.
        if state.level == 2:
            return

        if is_violent:
            state.strike_count += 1
            if state.strike_count >= self.trigger_count:
                state.level = 1 # Escalate to Orange
                state.label = f"{action_label.upper()} ({confidence:.0%})"
            else:
                state.label = f"suspicious ({state.strike_count}/{self.trigger_count})"
        else:
            # If safe, reset strikes but keep level at 0
            state.strike_count = 0
            state.level = 0
            state.label = f"{action_label} ({confidence:.0%})"

    def update_phase3(self, tracker_id, threat_detected, vlm_summary):
        """Called once by the slow Qwen model when analysis finishes."""
        state = self._ensure_exists(tracker_id)
        
        if threat_detected:
            # LOCK TO RED
            state.level = 2 
            state.label = "CONFIRMED THREAT"
            state.vlm_summary = vlm_summary
        else:
            # False alarm. Downgrade from Orange back to Green.
            state.level = 0
            state.strike_count = 0

    def get_ui_data(self, tracker_id):
        """Returns the color and text for the Visualizer."""
        if tracker_id not in self.states:
            return (0, 255, 0), "analyzing" # Default Green
            
        state = self.states[tracker_id]
        if state.level == 0: color = (0, 255, 0)       # Green
        elif state.level == 1: color = (0, 165, 255)   # Orange
        else: color = (0, 0, 255)                      # Red
        
        return color, state.label

    def cleanup(self, active_ids):
        """Removes memory of IDs that have left the camera for more than 10 seconds."""
        current_time = time.time()
        expired_ids = [
            tid for tid, state in self.states.items() 
            if tid not in active_ids and (current_time - state.last_seen > 10.0)
        ]
        for tid in expired_ids:
            del self.states[tid]