"""Apply the narrowly scoped GPUStack 2.2.3 Snap Docker pause workaround."""

import hashlib
from pathlib import Path

path = Path(
    "/usr/local/lib/python3.11/dist-packages/gpustack_runtime/deployer/docker.py"
)
expected = "49ffc5e4b39f8956bde2ffca9543da911a4054ea6efc52f9850f577333ee997f"
data = path.read_bytes()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("Unexpected GPUStack runtime source; review the patch before building")
old = '''        security_opt = [
            "no-new-privileges",
        ]
'''
new = '''        # Snap Docker cannot exec a privileged container with NNP enabled.
        # Retain NNP for every unprivileged pause container.
        security_opt = [] if privileged else ["no-new-privileges"]
'''
source = data.decode()
if source.count(old) != 1:
    raise SystemExit("Unexpected pause security configuration")
path.write_text(source.replace(old, new))
