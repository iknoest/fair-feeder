"""
Quick script to check if your Tapo C210 camera's ONVIF and RTSP are reachable.
Run: python check_onvif.py

Prerequisites:
  pip install onvif-zeep-async opencv-python

You need to have created a "Camera Account" in the Tapo app:
  Tapo app → Camera → Settings → Advanced Settings → Camera Account
"""
import asyncio
import sys
import os
import socket

# ── Config ──────────────────────────────────────────────────────────
# NOTE: Set these environment variables locally.
CAMERA_IP   = os.getenv('TAPO_IP',   '<YOUR_CAMERA_IP>')
CAMERA_USER = os.getenv('TAPO_USER', '<YOUR_CAMERA_USER>')
CAMERA_PASS = os.getenv('TAPO_PASS', '')
ONVIF_PORT  = int(os.getenv('TAPO_ONVIF_PORT', '2020'))
RTSP_PORT   = 554


def check_tcp(host, port, timeout=3):
    """Check if a TCP port is reachable."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


async def check_onvif():
    """Try to connect via ONVIF and list capabilities."""
    try:
        from onvif import ONVIFCamera
    except ImportError:
        print("❌ onvif-zeep-async not installed. Run: pip install onvif-zeep-async")
        return False

    # Find WSDL directory
    import onvif
    wsdl_dir = os.path.join(os.path.dirname(onvif.__file__), 'wsdl')
    if not os.path.isdir(wsdl_dir):
        print(f"❌ WSDL directory not found at: {wsdl_dir}")
        print("   You may need to download it from the onvif-zeep-async package.")
        return False

    print(f"   WSDL dir: {wsdl_dir}")

    try:
        cam = ONVIFCamera(
            CAMERA_IP,
            ONVIF_PORT,
            CAMERA_USER,
            CAMERA_PASS,
            wsdl_dir,
        )
        await cam.update_xaddrs()
        
        # Get device info
        device_service = await cam.create_devicemgmt_service()
        device_info = await device_service.GetDeviceInformation()
        print(f"   Manufacturer: {device_info.Manufacturer}")
        print(f"   Model:        {device_info.Model}")
        print(f"   Firmware:     {device_info.FirmwareVersion}")
        print(f"   Serial:       {device_info.SerialNumber}")

        # Check event service (needed for motion detection)
        try:
            event_service = await cam.create_events_service()
            caps = await event_service.GetServiceCapabilities()
            print(f"   Events supported: ✅")
            print(f"   PullPoint support: {'✅' if caps.WSPullPointSupport else '❌'}")
        except Exception as e:
            print(f"   Events: ❌ ({e})")

        await cam.close()
        return True

    except Exception as e:
        print(f"   ❌ ONVIF connection failed: {e}")
        return False


def check_rtsp():
    """Quick check if RTSP stream is readable."""
    try:
        import cv2
    except ImportError:
        print("   ❌ opencv-python not installed")
        return False

    url = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:{RTSP_PORT}/stream1"
    print(f"   URL: rtsp://{CAMERA_USER}:****@{CAMERA_IP}:{RTSP_PORT}/stream1")
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("   ❌ Could not open RTSP stream")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        h, w = frame.shape[:2]
        print(f"   ✅ Got frame: {w}x{h}")
        return True
    else:
        print("   ❌ Could not read frame")
        return False


async def main():
    print(f"{'='*55}")
    print(f"  Tapo C210 — ONVIF & RTSP Check")
    print(f"{'='*55}")
    print(f"  Camera IP: {CAMERA_IP}")
    print(f"  Username:  {CAMERA_USER}")
    print(f"  Password:  {'****' if CAMERA_PASS else '⚠️  EMPTY — set TAPO_PASS env var'}")
    print()

    # 1. TCP port checks
    print("1️⃣  TCP Port Reachability")
    onvif_ok = check_tcp(CAMERA_IP, ONVIF_PORT)
    print(f"   Port {ONVIF_PORT} (ONVIF): {'✅ Open' if onvif_ok else '❌ Closed/Unreachable'}")
    rtsp_ok = check_tcp(CAMERA_IP, RTSP_PORT)
    print(f"   Port {RTSP_PORT} (RTSP):  {'✅ Open' if rtsp_ok else '❌ Closed/Unreachable'}")
    print()

    # 2. ONVIF
    print("2️⃣  ONVIF Connection")
    if onvif_ok:
        onvif_result = await check_onvif()
    else:
        print("   ⏭️  Skipped (port not reachable)")
        onvif_result = False
    print()

    # 3. RTSP
    print("3️⃣  RTSP Stream")
    if rtsp_ok:
        rtsp_result = check_rtsp()
    else:
        print("   ⏭️  Skipped (port not reachable)")
        rtsp_result = False
    print()

    # Summary
    print(f"{'='*55}")
    print(f"  Summary")
    print(f"{'='*55}")
    if onvif_result and rtsp_result:
        print("  ✅ All good! Camera is ready for ONVIF motion recording.")
    elif rtsp_result and not onvif_result:
        print("  ⚠️  RTSP works but ONVIF failed.")
        print("  → Make sure you created a Camera Account in the Tapo app")
        print("  → Tapo app → Settings → Advanced Settings → Camera Account")
        print("  → Try port 2020 (default for Tapo ONVIF)")
    elif not rtsp_ok:
        print("  ❌ Camera not reachable. Check:")
        print("  → Is the camera on the same network?")
        print(f"  → Can you ping {CAMERA_IP}?")
        print("  → Is RTSP enabled in the Tapo app?")
    print()


if __name__ == "__main__":
    asyncio.run(main())
