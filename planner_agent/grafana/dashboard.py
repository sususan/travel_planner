# === Instrumentation (add near other imports at top of planner_agent.py) ===
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import threading

# Metrics - names chosen to be clear and Prometheus-friendly
ITINERARIES_CREATED = Counter(
    "planner_itineraries_created_total", "Total number of itinerary runs (successes).", ["status"]
)
PIPELINE_STEP_COUNT = Counter(
    "planner_pipeline_step_total", "Count of pipeline steps executed.", ["step", "status"]
)
GATE_FAILURES = Counter(
    "planner_gate_failures_total", "Count of gate failures observed.", ["gate"]
)
LAST_RUN_DURATION = Histogram(
    "planner_run_duration_seconds", "Duration of planner run in seconds (histogram)."
)
LAST_RUN_GAUGE = Gauge(
    "planner_last_run_timestamp", "Unix timestamp of last planner run."
)
PLANNER_ACTIVE = Gauge(
    "planner_active", "1 if planner process is up.", []
)
AVG_CO2 = Gauge(
    "planner_avg_co2_grams_per_itinerary", "Average estimated CO2 (grams) per itinerary."
)

# Start /metrics HTTP server once (choose port: 8000)
_METRICS_STARTED = False
def start_metrics_server(port: int = 8000):
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    # Run start_http_server in a daemon thread so it doesn't block
    def _start():
        start_http_server(port)
    t = threading.Thread(target=_start, daemon=True)
    t.start()
    _METRICS_STARTED = True
    PLANNER_ACTIVE.set(1)
