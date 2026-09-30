import re

def require_policy(log, device):
    if device == "cpu": return
    expected = "CPUAndGPU" if device == "gpu" else "ALL"
    policies = re.findall(r'measurement CoreML policy[^\n]*compute_units="([^"]+)"', log)
    assert policies and set(policies) == {expected}, f"missing or incorrect actual CoreML policy for {device}: {policies}"
