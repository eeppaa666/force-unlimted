from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene
from asyncio import sleep
import numpy as np
import os
from pathlib import Path

# 2.Try ~/.config/xr_teleoperate/
user_conf_dir = Path.home() / ".config" / "xr_teleoperate"
cert_path_user = user_conf_dir / "cert.pem"
key_path_user = user_conf_dir / "key.pem"
cert_file = None
key_file = None
if cert_path_user.exists() and key_path_user.exists():
    cert_file = cert_file or str(cert_path_user)
    key_file = key_file or str(key_path_user)
else:
    # 3.Fallback to package root (current logic)
    current_module_dir = Path(__file__).resolve().parent.parent.parent
    cert_file = cert_file or str(current_module_dir / "cert.pem")
    key_file = key_file or str(current_module_dir / "key.pem")
app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file)

@app.add_handler("CAMERA_MOVE")
async def handler(event, session: VuerSession):
    print("CAM", event)

@app.spawn(start=True)
async def main(session: VuerSession):
    session.set @ DefaultScene(frameloop="always")

    while True:
        await sleep(1.0)