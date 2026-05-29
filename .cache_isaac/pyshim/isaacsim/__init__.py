from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

try:
    from isaacsim.simulation_app import SimulationApp
except Exception:
    pass
