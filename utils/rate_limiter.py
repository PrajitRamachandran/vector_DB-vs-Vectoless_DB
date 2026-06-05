import time 
import threading

class RateLimiter:
  """
    Sliding window rate limiter.
    Ensures a minimum gap between requests to stay under RPM limits.

    Usage:
        limiter = RateLimiter(max_rpm=18, name="Mistral")
        limiter.wait()   # call this BEFORE every API request
        response = client.call(...)
    """
  
  def __init__(self,max_rpm:int,name:str=""):
    self.max_rpm = max_rpm
    self.min_gap = 60.0/max_rpm
    self.name = name
    self._last = 0.0
    self._lock = threading.Lock()

  def wait(self):
    """
        Blocks until enough time has passed since the last request.
        Prints a message only when it actually has to wait.
        """
    
    with self._lock:
      elapsed = time.monotonic()-self._last
      if elapsed<self.min_gap:
        wait_for = self.min_gap - elapsed
        print(f"   ⏳ {self.name} rate limit — waiting {wait_for:.1f}s...")
        time.sleep(wait_for)
      self._last = time.monotonic()

  def reset(self):
    """Call this if you want to start fresh (e.g. between test runs)."""
    self._last = 0.0
