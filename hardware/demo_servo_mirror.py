"""舵机手势镜像 MuJoCo — 纯被动回驱

人手拧舵机 → 编码器位置 → MuJoCo 关节 1:1 跟随。
参考 LeRobot SO100Leader 的 GroupSyncRead + 扭矩关闭模式。
"""
import os, sys, time, numpy as np, mujoco, mujoco.viewer

# --- 舵机 SDK ---
_servo_sdk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servo", "sdk")
sys.path.insert(0, _servo_sdk_dir)
from scservo_sdk import *  # noqa
import serial.tools.list_ports as _serial_ports

# --- 配置 ---
XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "so100_pick_place.xml")
SERVO_BAUD = 1000000
SERVO_DIR = -1
SERVO_MAP = {0: 1}  # joint_idx → servo_id
SERVO_TO_JOINT = {v: k for k, v in SERVO_MAP.items()}

# --- 扭矩控制（参考 LeRobot FeetechMotorsBus）---
def disable_torque(bus, sid):
    bus.write1ByteTxRx(sid, 40, 0)   # Torque_Enable = 0
    bus.write1ByteTxRx(sid, 48, 0)   # Lock = 0

def enable_torque(bus, sid):
    bus.write1ByteTxRx(sid, 48, 1)   # Lock = 1
    bus.write1ByteTxRx(sid, 40, 1)   # Torque_Enable = 1

# --- 位置映射 ---
def servo_to_rad(pos):
    """SCS225 位置 (0-1023) → MuJoCo 关节角 (rad)"""
    deg = pos * 300.0 / 1023.0 - 150.0
    return np.radians(deg * SERVO_DIR)

# --- 批量读取（参考 LeRobot sync_read）---
def read_servo_positions(bus, reader, sids):
    reader.clearParam()
    for sid in sids:
        reader.addParam(sid)
    reader.txRxPacket()
    return {sid: reader.getData(sid, 56, 2) for sid in sids}

# --- 主程序 ---
def main():
    # MuJoCo
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.dof_damping[:] = 3.0
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 舵机
    ports = list(_serial_ports.comports())
    if not ports:
        print("[SERVO] No COM port — running sim-only")
        servo_bus = None
    else:
        port = PortHandler(ports[0].device)
        port.setBaudRate(SERVO_BAUD)
        port.openPort()
        servo_bus = scscl(port)

        print("Scanning servos...")
        for j_idx, sid in SERVO_MAP.items():
            pos, result, _ = servo_bus.ReadPos(sid)
            if result == COMM_SUCCESS:
                print(f"  J{j_idx}=ID{sid} online, pos={pos}")
                disable_torque(servo_bus, sid)
                print(f"  J{j_idx}=ID{sid} torque OFF (free to backdrive)")
            else:
                print(f"  J{j_idx}=ID{sid} OFFLINE!")

    servo_ids = list(SERVO_MAP.values())

    print("\nMirror active. Twist the servo — MuJoCo follows.")
    print("Close viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0, -0.35, 0.10]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 270
        viewer.cam.elevation = -89

        while viewer.is_running():
            # 读舵机 → 写 MuJoCo（SCS 协议 v1 不支持 SyncRead，逐个读）
            if servo_bus is not None:
                for sid in servo_ids:
                    pos, result, _ = servo_bus.ReadPos(sid)
                    if result == COMM_SUCCESS:
                        joint_idx = SERVO_TO_JOINT.get(sid)
                        if joint_idx is not None:
                            data.ctrl[joint_idx] = servo_to_rad(pos)

            mujoco.mj_step(model, data)
            viewer.sync()

    # 清理
    if servo_bus is not None:
        for sid in servo_ids:
            enable_torque(servo_bus, sid)
            print(f"[SERVO] ID{sid} torque ON")
        port.closePort()
        print("[SERVO] Port closed.")
    print("[EXIT] Done.")

if __name__ == "__main__":
    main()
