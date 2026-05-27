import time
import argparse
import json
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.g1_episode_client import G1EpisodeClient
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_START_REQUEST = False  # Request recording start
RECORD_STOP_REQUEST  = False  # Request recording stop/save
RECORD_TRANSFER_REQUEST = False  # Request pulling completed G1 episodes back to this machine
RECORD_TRANSFER_RUNNING = False  # True while completed G1 episodes are being transferred
RECORD_START_TO_HOME_TIME = 2.0     # seconds, only used right after pressing s
RECORD_START_RETURN_TIME = 2.0      # seconds, only used right after pressing s
RECORD_START_ARM_SMOOTH_TIME = 3.0  # seconds, used by current record-start smoothing path
RECORD_START_ARM_MAX_SPEED = 0.18   # rad/s, used by current record-start smoothing path
RECORD_DEBUG_POST_RECORD_SECONDS = 5.0
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_START_REQUEST
#                    False       True          False        False         False                  False
#   RECORD_STOP_REQUEST
#                    False       False         False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: press s to request start, press t to request stop/save.
#  --> auto  : Auto-transition after saving data.

class StartupGuardError(RuntimeError):
    pass

class SafetyCheckError(RuntimeError):
    pass

_KEY_TO_TELEOP_ACTION = {
    'r': 'start',
    'q': 'stop',
    's': 'record_start',
    't': 'record_stop',
    'p': 'record_transfer',
}

def request_teleop_action(action: str, source: str = "Keyboard"):
    """Unified teleop control: keyboard, IPC (via on_press), and XR controller shortcuts."""
    global STOP, START, RECORD_START_REQUEST, RECORD_STOP_REQUEST, RECORD_TRANSFER_REQUEST
    if action == 'start':
        START = True
        logger_mp.info("[%s] start tracking requested.", source)
    elif action == 'stop':
        START = False
        STOP = True
        logger_mp.info("[%s] safe shutdown requested.", source)
    elif action == 'record_start':
        if START:
            RECORD_START_REQUEST = True
            logger_mp.info("[%s] record start requested.", source)
        else:
            logger_mp.warning("[%s] record start ignored: tracking not started yet (press r / Left A first).", source)
    elif action == 'record_stop':
        if START:
            RECORD_STOP_REQUEST = True
            logger_mp.info("[%s] record stop/save requested.", source)
        else:
            logger_mp.warning("[%s] record stop ignored: tracking not started yet.", source)
    elif action == 'record_transfer':
        RECORD_TRANSFER_REQUEST = True
        logger_mp.info("[%s] transfer completed G1 episodes requested.", source)
    else:
        logger_mp.warning("[%s] unknown teleop action: %s", source, action)

def on_press(key):
    key = str(key).lower()
    action = _KEY_TO_TELEOP_ACTION.get(key)
    if action:
        request_teleop_action(action, "Keyboard")
    else:
        logger_mp.warning("[Keyboard] %s was pressed, but no action is defined for this key.", key)

class _ControllerShortcutState:
    """Edge detection and Right-B short (record start) vs long (transfer) press handling."""

    def __init__(self):
        self._prev_buttons = {}
        self._right_b_pressed_at = None
        self._right_b_long_press_fired = False

    def reset_right_b(self):
        self._right_b_pressed_at = None
        self._right_b_long_press_fired = False

    def _rising_edge(self, name: str, pressed: bool) -> bool:
        prev = self._prev_buttons.get(name, False)
        self._prev_buttons[name] = pressed
        return pressed and not prev

    def update(self, tele_data, *, allow_record_actions: bool, long_press_s: float):
        if self._rising_edge('left_a', tele_data.left_ctrl_aButton):
            request_teleop_action('start', 'Controller')
        if self._rising_edge('right_a', tele_data.right_ctrl_aButton):
            request_teleop_action('stop', 'Controller')
        if allow_record_actions and self._rising_edge('left_b', tele_data.left_ctrl_bButton):
            request_teleop_action('record_stop', 'Controller')

        right_b = tele_data.right_ctrl_bButton
        now = time.monotonic()
        if right_b:
            if self._right_b_pressed_at is None:
                self._right_b_pressed_at = now
                self._right_b_long_press_fired = False
            elif (
                allow_record_actions
                and not self._right_b_long_press_fired
                and (now - self._right_b_pressed_at) >= long_press_s
            ):
                request_teleop_action('record_transfer', 'Controller')
                logger_mp.info(
                    "[Controller] right B held %.1fs: episode transfer (p) requested.",
                    long_press_s,
                )
                self._right_b_long_press_fired = True
        elif self._right_b_pressed_at is not None:
            held_s = now - self._right_b_pressed_at
            if allow_record_actions and not self._right_b_long_press_fired and held_s < long_press_s:
                request_teleop_action('record_start', 'Controller')
                logger_mp.info(
                    "[Controller] right B short press (%.2fs): record start (s) requested.",
                    held_s,
                )
            self.reset_right_b()

def _controller_shortcuts_enabled(args) -> bool:
    return args.input_mode == 'controller' and not args.no_controller_shortcuts

def _log_teleop_control_hints(args):
    logger_mp.info("----------------------------------------------------------------")
    logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
    if args.record:
        logger_mp.info("🟡  Press [s] to START recording, [t] to STOP and SAVE recording, [p] to PULL completed G1 episodes.")
    else:
        logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
    logger_mp.info("🔴  Press [q] to stop and exit the program.")
    if _controller_shortcuts_enabled(args):
        logger_mp.info("🎮  Controller shortcuts (same as keys above):")
        logger_mp.info("     Left A (X) → start (r)  |  Right A → quit (q)")
        logger_mp.info("     Right B short release → record START (s)  |  Left B (Y) → record STOP (t)")
        logger_mp.info(
            "     Right B hold ≥%.1fs → pull G1 episodes (p)",
            args.controller_right_b_long_press_s,
        )
    logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")

def _listen_keyboard_safe():
    try:
        listen_keyboard(on_press=on_press, until=None, sequential=True)
    except ValueError as exc:
        if "closed file" in str(exc):
            logger_mp.debug("[Keyboard] listener exited after stdin was closed.")
        else:
            raise
    except Exception as exc:
        if STOP:
            logger_mp.debug("[Keyboard] listener exited during shutdown: %s", exc)
        else:
            logger_mp.exception("[Keyboard] listener failed.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY, RECORD_TRANSFER_RUNNING
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "RECORD_TRANSFER_RUNNING": RECORD_TRANSFER_RUNNING,
    }

def _clamp01(value):
    return max(0.0, min(1.0, value))

def _controller_grip_value(pressed, analog_value, open_value=10.0):
    """Convert TeleData's trigger-like values to grip: 0.0=open, 1.0=closed."""
    _ = pressed
    try:
        value = float(analog_value)
        if value == value:  # not NaN
            return _clamp01(1.0 - value / open_value)
    except (TypeError, ValueError):
        pass
    return 0.0

def _dex3_controller_grip(tele_data, side):
    return _controller_grip_value(
        getattr(tele_data, f"{side}_ctrl_trigger", False),
        getattr(tele_data, f"{side}_ctrl_triggerValue", 10.0),
    )

def _controller_squeeze_value(pressed, analog_value):
    """Convert controller side-grip values to 0.0=open, 1.0=closed."""
    try:
        value = float(analog_value)
        if value == value:  # not NaN
            return _clamp01(value)
    except (TypeError, ValueError):
        pass
    return 1.0 if pressed else 0.0

def _image_timestamp_ns(img):
    if img is None:
        return None
    return getattr(img, "capture_time_ns", None) or getattr(img, "recv_time_ns", None)

def _image_within_time_window(img, target_time_ns, max_delta_s):
    if img is None:
        return False
    if target_time_ns is None:
        return True
    img_time_ns = _image_timestamp_ns(img)
    if img_time_ns is None:
        return False
    return abs(img_time_ns - target_time_ns) <= int(max_delta_s * 1_000_000_000)

def _dex3_controller_pinch(tele_data, side):
    return _controller_squeeze_value(
        getattr(tele_data, f"{side}_ctrl_squeeze", False),
        getattr(tele_data, f"{side}_ctrl_squeezeValue", 0.0),
    )

# Dex3 q order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1.
DEX3_LEFT_OPEN_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEX3_RIGHT_OPEN_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEX3_LEFT_CLOSED_Q = [0.0, 0.75, 1.35, -1.15, -1.35, -1.15, -1.35]
DEX3_RIGHT_CLOSED_Q = [0.0, -0.75, -1.35, 1.15, 1.35, 1.15, 1.35]
DEX3_PINCH_THUMB_OPPOSITION_Q = -0.55
DEX3_LEFT_PINCH_Q = [DEX3_PINCH_THUMB_OPPOSITION_Q, 0.55, 0.75, 0.0, 0.0, -1.15, -0.90]
DEX3_RIGHT_PINCH_Q = [DEX3_PINCH_THUMB_OPPOSITION_Q, -0.55, -0.75, 0.0, 0.0, 1.15, 0.90]

def _lerp_array(open_q, closed_q, grip):
    grip = _clamp01(grip)
    return [open_q[i] + (closed_q[i] - open_q[i]) * grip for i in range(len(open_q))]

def _make_dex3_controller_q_target(left_grip, right_grip):
    left_q = _lerp_array(DEX3_LEFT_OPEN_Q, DEX3_LEFT_CLOSED_Q, left_grip)
    right_q = _lerp_array(DEX3_RIGHT_OPEN_Q, DEX3_RIGHT_CLOSED_Q, right_grip)
    return left_q + right_q

def _make_dex3_true_controller_q_target(left_grip, right_grip, left_pinch, right_pinch):
    left_q = _lerp_array(
        DEX3_LEFT_OPEN_Q,
        DEX3_LEFT_PINCH_Q if left_pinch > 1e-4 else DEX3_LEFT_CLOSED_Q,
        left_pinch if left_pinch > 1e-4 else left_grip,
    )
    right_q = _lerp_array(
        DEX3_RIGHT_OPEN_Q,
        DEX3_RIGHT_PINCH_Q if right_pinch > 1e-4 else DEX3_RIGHT_CLOSED_Q,
        right_pinch if right_pinch > 1e-4 else right_grip,
    )
    return left_q + right_q

def _dex3_q_to_grip(q, open_q, closed_q):
    grip_values = []
    for q_i, open_i, closed_i in zip(q, open_q, closed_q):
        delta = closed_i - open_i
        if abs(delta) > 1e-6:
            grip_values.append(_clamp01((q_i - open_i) / delta))
    return sum(grip_values) / len(grip_values) if grip_values else 0.0

def _smoothstep(alpha):
    alpha = _clamp01(alpha)
    return alpha * alpha * (3.0 - 2.0 * alpha)

def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value

def _max_abs(value):
    array = np.asarray(value, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0

def _record_debug(debug_file, event, **payload):
    if debug_file is None:
        return
    row = {
        "time": time.time(),
        "event": event,
    }
    row.update(payload)
    debug_file.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")

def _create_episode_worker(recorder, done_event, result):
    started_at = time.time()
    try:
        ok = recorder.create_episode()
        error = None
    except Exception as exc:
        ok = False
        error = repr(exc)
        logger_mp.exception("Failed to create episode in background thread.")
    result["ok"] = ok
    result["elapsed"] = time.time() - started_at
    result["error"] = error
    done_event.set()

def _engage_first_arm_target(arm_ctrl, arm_ik, tele_data, max_joint_delta, transition_time, frequency):
    """Reject unsafe first targets, otherwise ease from current arm q to the first IK q."""
    global STOP

    current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float)
    current_dq = np.asarray(arm_ctrl.get_current_dual_arm_dq(), dtype=float)
    zero_tauff = np.zeros_like(current_q)
    arm_ctrl.ctrl_dual_arm(current_q, zero_tauff)

    sol_q, sol_tauff = arm_ik.solve_ik(
        tele_data.left_wrist_pose,
        tele_data.right_wrist_pose,
        current_q,
        current_dq,
    )
    target_q = np.asarray(sol_q, dtype=float)
    target_tauff = np.asarray(sol_tauff, dtype=float)
    if target_q.shape != current_q.shape:
        raise StartupGuardError(f"Startup IK q shape mismatch: current={current_q.shape}, target={target_q.shape}")
    if target_tauff.shape != current_q.shape:
        raise StartupGuardError(f"Startup IK tau shape mismatch: current={current_q.shape}, tau={target_tauff.shape}")

    delta_q = target_q - current_q
    if not np.all(np.isfinite(delta_q)) or not np.all(np.isfinite(target_tauff)):
        raise StartupGuardError("Startup IK target contains non-finite joint values.")

    abs_delta_q = np.abs(delta_q)
    max_delta = float(np.max(abs_delta_q)) if abs_delta_q.size else 0.0
    max_delta_idx = int(np.argmax(abs_delta_q)) if abs_delta_q.size else -1
    logger_mp.info(
        f"[Startup Guard] first target max joint delta={max_delta:.3f} rad "
        f"(joint index {max_delta_idx}), limit={max_joint_delta:.3f} rad"
    )

    if max_joint_delta > 0.0 and max_delta > max_joint_delta:
        raise StartupGuardError(
            "[Startup Guard] Rejected first arm target because joint delta is too large: "
            f"max_delta={max_delta:.3f} rad at joint index {max_delta_idx}, "
            f"limit={max_joint_delta:.3f} rad. Please realign the XR pose and robot initial pose."
        )

    transition_time = max(0.0, float(transition_time))
    frequency = max(1.0, float(frequency))
    if transition_time <= 0.0:
        arm_ctrl.ctrl_dual_arm(target_q, target_tauff)
        return

    steps = max(1, int(transition_time * frequency))
    logger_mp.info(
        f"[Startup Guard] easing to first target over {transition_time:.2f}s "
        f"({steps} steps)."
    )
    for step in range(1, steps + 1):
        if STOP:
            logger_mp.warning("[Startup Guard] transition interrupted by stop signal.")
            return
        alpha = _smoothstep(step / steps)
        q_cmd = current_q + delta_q * alpha
        tauff_cmd = target_tauff * alpha
        arm_ctrl.ctrl_dual_arm(q_cmd, tauff_cmd)
        time.sleep(1.0 / frequency)

def _ease_arm_to_home(arm_ctrl, transition_time, frequency, label="Shutdown Guard"):
    return _ease_arm_to_q(
        arm_ctrl=arm_ctrl,
        target_q=np.zeros_like(np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float)),
        transition_time=transition_time,
        frequency=frequency,
        label=label,
    )

def _ease_arm_to_q(arm_ctrl, target_q, transition_time, frequency, label):
    current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float)
    target_q = np.asarray(target_q, dtype=float)
    if target_q.shape != current_q.shape:
        raise SafetyCheckError(f"{label} target shape mismatch: current={current_q.shape}, target={target_q.shape}")
    if not np.all(np.isfinite(target_q)):
        raise SafetyCheckError(f"{label} target contains NaN/Inf.")
    zero_tauff = np.zeros_like(current_q)

    transition_time = max(0.0, float(transition_time))
    frequency = max(1.0, float(frequency))
    if transition_time <= 0.0:
        arm_ctrl.ctrl_dual_arm(target_q, zero_tauff)
        return

    steps = max(1, int(transition_time * frequency))
    logger_mp.info(
        f"[{label}] easing arms over {transition_time:.2f}s "
        f"({steps} steps)."
    )
    delta_q = target_q - current_q
    for step in range(1, steps + 1):
        alpha = _smoothstep(step / steps)
        q_cmd = current_q + delta_q * alpha
        arm_ctrl.ctrl_dual_arm(q_cmd, zero_tauff)
        time.sleep(1.0 / frequency)

def _open_end_effectors_for_shutdown(args, frequency, dex3_direct_q_target_array=None, left_gripper_value=None, right_gripper_value=None):
    frequency = max(1.0, float(frequency))
    hold_time = 1.0
    steps = max(1, int(hold_time * frequency))

    if args.ee in ("dex3", "dex3_true") and args.input_mode == "controller" and dex3_direct_q_target_array is not None:
        open_q = DEX3_LEFT_OPEN_Q + DEX3_RIGHT_OPEN_Q
        logger_mp.info("[Shutdown Guard] opening Dex3 hands over %.2fs.", hold_time)
        for _ in range(steps):
            with dex3_direct_q_target_array.get_lock():
                dex3_direct_q_target_array[:] = open_q
            time.sleep(1.0 / frequency)
        return True

    if args.ee == "dex1" and left_gripper_value is not None and right_gripper_value is not None:
        logger_mp.info("[Shutdown Guard] opening Dex1 grippers over %.2fs.", hold_time)
        for _ in range(steps):
            with left_gripper_value.get_lock():
                left_gripper_value.value = 7.0
            with right_gripper_value.get_lock():
                right_gripper_value.value = 7.0
            time.sleep(1.0 / frequency)
        return True

    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--no-controller-shortcuts', action='store_true', help='Disable XR controller button mapping to teleop actions (keyboard/IPC still work).')
    parser.add_argument('--controller-right-b-long-press-s', type=float, default=2.0, help='Right B hold duration (seconds) to trigger episode transfer (p) in controller mode.')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'dex3_true', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--startup-max-joint-delta', type=float, default=1.2, help='Abort startup if any first IK target joint delta exceeds this many radians. Set <=0 to disable.')
    parser.add_argument('--startup-transition-time', type=float, default=3.0, help='Seconds used to ease from current arm joints to the first IK target after startup guard passes.')
    parser.add_argument('--ready-transition-time', type=float, default=5.0, help='Seconds used to ease arms to the robot home/initial pose before waiting for r.')
    parser.add_argument('--shutdown-transition-time', type=float, default=5.0, help='Seconds used to ease from current arm joints to home before exiting.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    parser.add_argument('--skip-ready-pose', action='store_true', help='Do not move arms to home/initial pose before waiting for r.')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--record_debug_true', action='store_true', help='Write record_debug_*.jsonl logs for troubleshooting recording/start jitter.')
    parser.add_argument('--record-rerun-log', action='store_true', help='Enable Rerun online visualization for recorded episodes. Disabled by default to avoid control jitter on real robots.')
    parser.add_argument('--record-backend', type=str, choices=['g1', 'local'], default='g1', help='Record episodes on G1 tmp_data or locally on this machine.')
    parser.add_argument('--g1-record-port', type=int, default=60010, help='G1-local episode recorder command port.')
    parser.add_argument('--record-transfer-ip', type=str, default=None, help='G1 IP used for pulling finished episodes back. Defaults to --img-server-ip.')
    parser.add_argument('--record-transfer-user', type=str, default=None, help='SSH user for pulling G1 episodes. Defaults to current user.')
    parser.add_argument('--record-transfer-ssh-port', type=int, default=22, help='SSH port for pulling G1 episodes.')
    parser.add_argument('--record-transfer-bwlimit-kbps', type=int, default=4000, help='Rsync --bwlimit value for episode pullback in KiB/s. Set 0 to disable.')
    parser.add_argument('--g1-record-max-pending-items', type=int, default=0, help='Maximum queued G1 add_item commands; 0 keeps all frames and disables realtime dropping.')
    parser.add_argument('--depth-sync-offset-s', type=float, default=0.0, help='Recording-only depth time offset. Positive means choose an older depth frame than the RGB frame.')
    parser.add_argument('--depth-sync-max-delta-s', type=float, default=0.12, help='Maximum allowed time distance when matching depth to RGB; about 1-2 depth frames at 15 Hz.')
    parser.add_argument('--depth-sync-wait-s', type=float, default=0.01, help='Recording-only wait time for a near depth frame to arrive.')
    parser.add_argument('--task-dir', type = str, default = './utils/dex3_true/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick_cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    if args.frequency <= 0.0:
        parser.error("--frequency must be greater than 0.")
    if args.startup_transition_time < 0.0:
        parser.error("--startup-transition-time must be greater than or equal to 0.")
    if args.ready_transition_time < 0.0:
        parser.error("--ready-transition-time must be greater than or equal to 0.")
    if args.shutdown_transition_time < 0.0:
        parser.error("--shutdown-transition-time must be greater than or equal to 0.")
    if args.depth_sync_max_delta_s < 0.0:
        parser.error("--depth-sync-max-delta-s must be greater than or equal to 0.")
    if args.depth_sync_wait_s < 0.0:
        parser.error("--depth-sync-wait-s must be greater than or equal to 0.")
    if args.record_transfer_bwlimit_kbps < 0:
        parser.error("--record-transfer-bwlimit-kbps must be greater than or equal to 0.")
    if args.g1_record_max_pending_items < 0:
        parser.error("--g1-record-max-pending-items must be greater than or equal to 0.")
    if args.controller_right_b_long_press_s <= 0.0:
        parser.error("--controller-right-b-long-press-s must be greater than 0.")
    logger_mp.info(f"args: {args}")
    use_g1_record = args.record and args.record_backend == 'g1'

    record_debug_file = None
    record_debug_path = None
    if args.record and args.record_debug_true:
        record_debug_path = os.path.join(parent_dir, f"record_debug_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        record_debug_file = open(record_debug_path, "a", buffering=1, encoding="utf-8")
        logger_mp.info(f"[Record Debug] writing debug log to {record_debug_path}")
        _record_debug(record_debug_file, "program_start", args=vars(args), debug_path=record_debug_path)

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=_listen_keyboard_safe, daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(
            host=args.img_server_ip,
            request_bgr=True,
            subscribe_topics=[] if use_g1_record else None,
        )
        camera_config = img_client.get_cam_config()
        depth_camera_cfg = camera_config.get('depth_camera', {})
        depth_zmq_enabled = depth_camera_cfg.get('enable_zmq', False)
        if args.record and depth_zmq_enabled and not use_g1_record:
            logger_mp.info(
                f"[Image Record] depth_camera ZMQ enabled on port "
                f"{depth_camera_cfg.get('zmq_port')}; saving frames as depths/depth_0."
            )
            logger_mp.info(
                f"[Image Record] depth sync offset={args.depth_sync_offset_s:.3f}s, "
                f"max_delta={args.depth_sync_max_delta_s:.3f}s, "
                f"wait={args.depth_sync_wait_s:.3f}s."
            )
        last_depth_img = None
        last_depth_fallback_log_t = 0.0
        last_depth_skip_log_t = 0.0
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            if status == 0:
                logger_mp.info(f"Enter debug mode: Success, status={status}, result={result}")
            else:
                raise SafetyCheckError(f"Enter debug mode failed, aborting for safety: status={status}, result={result}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3" or args.ee == "dex3_true":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            dex3_direct_q_target_array = Array('d', 14, lock = True) if args.input_mode == "controller" else None
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                          direct_q_target_array_in=dex3_direct_q_target_array)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            record_task_dir = os.path.join(args.task_dir, args.task_name)
            if use_g1_record:
                recorder = G1EpisodeClient(
                    host=args.img_server_ip,
                    task_dir=record_task_dir,
                    task_goal=args.task_goal,
                    task_desc=args.task_desc,
                    task_steps=args.task_steps,
                    frequency=args.frequency,
                    recorder_port=args.g1_record_port,
                    transfer_ip=args.record_transfer_ip or args.img_server_ip,
                    transfer_user=args.record_transfer_user,
                    transfer_ssh_port=args.record_transfer_ssh_port,
                    transfer_bwlimit_kbps=args.record_transfer_bwlimit_kbps,
                    max_pending_add_items=args.g1_record_max_pending_items,
                )
            else:
                recorder = EpisodeWriter(task_dir = record_task_dir,
                                         task_goal = args.task_goal,
                                         task_desc = args.task_desc,
                                         task_steps = args.task_steps,
                                         frequency = args.frequency,
                                         rerun_log = args.record_rerun_log and not args.headless)

        if not args.skip_ready_pose:
            _ease_arm_to_home(arm_ctrl, args.ready_transition_time, args.frequency, label="Ready Pose")

        _log_teleop_control_hints(args)
        controller_shortcut_state = (
            _ControllerShortcutState() if _controller_shortcuts_enabled(args) else None
        )
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            if controller_shortcut_state is not None:
                wait_tele_data = tv_wrapper.get_tele_data()
                controller_shortcut_state.update(
                    wait_tele_data,
                    allow_record_actions=False,
                    long_press_s=args.controller_right_b_long_press_s,
                )
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        if controller_shortcut_state is not None:
            controller_shortcut_state.reset_right_b()

        if STOP:
            logger_mp.info("Stop requested before tracking started.")
            sys.exit(0)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        startup_tele_data = tv_wrapper.get_tele_data()
        _engage_first_arm_target(
            arm_ctrl=arm_ctrl,
            arm_ik=arm_ik,
            tele_data=startup_tele_data,
            max_joint_delta=args.startup_max_joint_delta,
            transition_time=args.startup_transition_time,
            frequency=args.frequency,
        )
        arm_ctrl.speed_gradual_max()
        record_start_smooth_q = None
        record_start_smooth_tauff = None
        record_start_smooth_started_at = 0.0
        record_create_thread = None
        record_create_done = None
        record_create_result = None
        record_stop_after_create = False
        record_start_after_ready = False
        record_transfer_after_ready = False
        last_arm_cmd_q = None
        last_arm_cmd_tauff = None
        record_debug_frames_remaining = 0
        _record_debug(record_debug_file, "tracking_started")
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            if record_create_thread is not None and record_create_done is not None and record_create_done.is_set():
                record_create_thread.join(timeout=0.0)
                create_ok = bool(record_create_result.get("ok", False))
                create_episode_elapsed = record_create_result.get("elapsed")
                create_error = record_create_result.get("error")
                _record_debug(
                    record_debug_file,
                    "create_episode_end",
                    ok=create_ok,
                    elapsed=create_episode_elapsed,
                    error=create_error,
                    record_running_before=RECORD_RUNNING,
                )
                if create_ok:
                    RECORD_RUNNING = True
                    if record_debug_file is not None:
                        record_debug_frames_remaining = int(max(1.0, RECORD_DEBUG_POST_RECORD_SECONDS) * args.frequency)
                    logger_mp.info(
                        f"[Record Guard] background create_episode finished in "
                        f"{create_episode_elapsed:.2f}s. Recording started."
                    )
                    if record_stop_after_create:
                        RECORD_STOP_REQUEST = True
                        record_stop_after_create = False
                        logger_mp.info("[Record Guard] queued stop request will be handled now that recording has started.")
                else:
                    record_start_smooth_q = None
                    record_start_smooth_tauff = None
                    record_stop_after_create = False
                    record_start_after_ready = False
                    logger_mp.error(f"Failed to create episode. Recording not started. error={create_error}")
                record_create_thread = None
                record_create_done = None
                record_create_result = None

            # get image
            depth_img = None
            if camera_config['head_camera']['enable_zmq']:
                if (args.record and not use_g1_record) or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record and not use_g1_record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record and not use_g1_record:
                    right_wrist_img = img_client.get_right_wrist_frame()
            if depth_zmq_enabled:
                if args.record and not use_g1_record:
                    head_time_ns = None
                    depth_target_time_ns = None
                    if 'head_img' in locals() and head_img is not None:
                        head_time_ns = getattr(head_img, "capture_time_ns", None) or getattr(head_img, "recv_time_ns", None)
                    if head_time_ns is not None:
                        depth_target_time_ns = head_time_ns - int(args.depth_sync_offset_s * 1_000_000_000)
                        depth_wait_deadline = time.monotonic() + args.depth_sync_wait_s
                        while True:
                            depth_img = img_client.get_depth_frame_near(
                                depth_target_time_ns,
                                max_delta_s=args.depth_sync_max_delta_s,
                            )
                            if depth_img is not None or time.monotonic() >= depth_wait_deadline:
                                break
                            time.sleep(0.002)
                    if depth_img is None:
                        latest_depth_img = img_client.get_depth_frame()
                        if _image_within_time_window(latest_depth_img, depth_target_time_ns, args.depth_sync_max_delta_s):
                            depth_img = latest_depth_img
                        elif _image_within_time_window(last_depth_img, depth_target_time_ns, args.depth_sync_max_delta_s):
                            depth_img = last_depth_img
                        now_for_log = time.monotonic()
                        if now_for_log - last_depth_fallback_log_t > 2.0:
                            logger_mp.warning(
                                "[Image Record] No nearest depth hit; tried latest/last within sync window."
                            )
                            last_depth_fallback_log_t = now_for_log
                    if depth_img is not None:
                        last_depth_img = depth_img

            pending_record_start = False
            pending_record_stop = False
            pending_record_transfer = False
            record_create_pending = (
                record_create_thread is not None
                and record_create_done is not None
                and not record_create_done.is_set()
            )
            record_start_requested = False
            record_stop_requested = False
            record_transfer_requested = False
            if args.record and use_g1_record and 'recorder' in locals():
                RECORD_TRANSFER_RUNNING = bool(recorder.is_transferring())
            else:
                RECORD_TRANSFER_RUNNING = False
            if args.record:
                record_start_requested = RECORD_START_REQUEST
                record_stop_requested = RECORD_STOP_REQUEST
                record_transfer_requested = RECORD_TRANSFER_REQUEST
                RECORD_START_REQUEST = False
                RECORD_STOP_REQUEST = False
                RECORD_TRANSFER_REQUEST = False
                if (
                    record_start_after_ready
                    and not RECORD_RUNNING
                    and not record_create_pending
                    and not record_transfer_requested
                    and not record_transfer_after_ready
                    and recorder.is_ready()
                ):
                    record_start_requested = True
                    record_start_after_ready = False
                    logger_mp.info("[Record Guard] queued start request is running now.")
                if (
                    record_transfer_after_ready
                    and use_g1_record
                    and not RECORD_RUNNING
                    and not record_create_pending
                    and not recorder.is_transferring()
                    and recorder.pending_transfer_count() > 0
                ):
                    pending_record_transfer = True
                    record_transfer_after_ready = False
                    logger_mp.info("[G1EpisodeClient] queued p-transfer is running now.")

            if args.record and record_transfer_requested:
                if not use_g1_record:
                    logger_mp.warning("[G1EpisodeClient] p ignored: current record backend is local, not G1.")
                elif RECORD_RUNNING or record_create_pending or pending_record_start or pending_record_stop:
                    logger_mp.warning("[G1EpisodeClient] p ignored while recording/starting/stopping. Press t first, then p after save is queued.")
                elif recorder.is_transferring():
                    logger_mp.warning("[G1EpisodeClient] p ignored: episode transfer is already running.")
                elif recorder.pending_transfer_count() <= 0:
                    if not recorder.is_ready():
                        record_transfer_after_ready = True
                        logger_mp.info("[G1EpisodeClient] p queued: recorder is finishing stop; transfer will start once the episode is ready.")
                    else:
                        logger_mp.warning("[G1EpisodeClient] p pressed, but there are no completed episodes waiting for transfer.")
                else:
                    pending_record_transfer = True

            if args.record and record_start_requested:
                if RECORD_RUNNING:
                    _record_debug(record_debug_file, "record_start_ignored_already_running")
                    logger_mp.warning("[Record Guard] already recording; ignored duplicate s press.")
                elif record_create_pending:
                    _record_debug(record_debug_file, "record_start_ignored_create_pending")
                    logger_mp.warning("[Record Guard] create_episode is still pending; ignored duplicate s press.")
                elif pending_record_transfer:
                    _record_debug(record_debug_file, "record_start_ignored_transfer_requested")
                    logger_mp.warning("[Record Guard] s ignored: p-transfer was requested in this cycle. Wait for it to finish.")
                elif record_transfer_after_ready:
                    _record_debug(record_debug_file, "record_start_ignored_transfer_queued")
                    logger_mp.warning("[Record Guard] s ignored: p-transfer is queued. Wait for it to finish.")
                elif use_g1_record and recorder.is_transferring():
                    _record_debug(record_debug_file, "record_start_ignored_transfer_running")
                    logger_mp.warning("[Record Guard] s ignored: G1 episode transfer is running. Wait for p-transfer to finish.")
                elif not recorder.is_ready():
                    if not record_start_after_ready:
                        _record_debug(record_debug_file, "record_start_queued_not_ready")
                        logger_mp.info("[Record Guard] recorder is finishing the previous stop; queued s and will start as soon as it is ready.")
                    else:
                        _record_debug(record_debug_file, "record_start_ignored_already_queued")
                        logger_mp.warning("[Record Guard] start is already queued; ignored duplicate s press.")
                    record_start_after_ready = True
                else:
                    pending_record_start = True
                    record_start_after_ready = False
                    if last_arm_cmd_q is not None:
                        record_start_smooth_q = last_arm_cmd_q.copy()
                    else:
                        record_start_smooth_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float)
                    record_start_smooth_tauff = last_arm_cmd_tauff.copy() if last_arm_cmd_tauff is not None else None
                    record_start_smooth_started_at = time.time()
                    if record_debug_file is not None:
                        _record_debug(
                            record_debug_file,
                            "record_start_request",
                            record_running=RECORD_RUNNING,
                            current_q_max_abs=_max_abs(arm_ctrl.get_current_dual_arm_q()),
                            last_cmd_q_max_abs=_max_abs(record_start_smooth_q),
                            last_cmd_tauff_max_abs=_max_abs(record_start_smooth_tauff) if record_start_smooth_tauff is not None else None,
                        )
                    logger_mp.info(
                        f"[Record Guard] smoothing arm target after s "
                        f"at {RECORD_START_ARM_MAX_SPEED:.2f} rad/s."
                    )

            if args.record and record_stop_requested:
                if RECORD_RUNNING:
                    pending_record_stop = True
                    _record_debug(record_debug_file, "record_stop_request", record_running=RECORD_RUNNING)
                elif record_create_pending or pending_record_start:
                    if not record_stop_after_create:
                        _record_debug(record_debug_file, "record_stop_queued_create_pending")
                        logger_mp.info("[Record Guard] stop requested while recording is starting; will save after create_episode finishes.")
                    else:
                        _record_debug(record_debug_file, "record_stop_ignored_already_queued")
                        logger_mp.warning("[Record Guard] stop is already queued; ignored duplicate t press.")
                    record_stop_after_create = True
                else:
                    _record_debug(record_debug_file, "record_stop_ignored_not_running")
                    logger_mp.warning("[Record Guard] not recording; ignored duplicate t press.")

            pending_record_command = pending_record_start or pending_record_stop or pending_record_transfer

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "dex3_true" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif (args.ee == "dex3" or args.ee == "dex3_true") and args.input_mode == "controller":
                target_left_grip = _dex3_controller_grip(tele_data, "left")
                target_right_grip = _dex3_controller_grip(tele_data, "right")
                prev_left_grip, prev_right_grip = getattr(tv_wrapper, "_dex3_controller_grip", (0.0, 0.0))
                grip_alpha = 0.35
                left_dex3_grip = prev_left_grip + (target_left_grip - prev_left_grip) * grip_alpha
                right_dex3_grip = prev_right_grip + (target_right_grip - prev_right_grip) * grip_alpha
                tv_wrapper._dex3_controller_grip = (left_dex3_grip, right_dex3_grip)

                dex3_q_target = _make_dex3_controller_q_target(left_dex3_grip, right_dex3_grip)
                if args.ee == "dex3_true":
                    target_left_pinch = _dex3_controller_pinch(tele_data, "left")
                    target_right_pinch = _dex3_controller_pinch(tele_data, "right")
                    prev_left_pinch, prev_right_pinch = getattr(tv_wrapper, "_dex3_controller_pinch", (0.0, 0.0))
                    left_dex3_pinch = prev_left_pinch + (target_left_pinch - prev_left_pinch) * grip_alpha
                    right_dex3_pinch = prev_right_pinch + (target_right_pinch - prev_right_pinch) * grip_alpha
                    tv_wrapper._dex3_controller_pinch = (left_dex3_pinch, right_dex3_pinch)
                    dex3_q_target = _make_dex3_true_controller_q_target(
                        left_dex3_grip,
                        right_dex3_grip,
                        left_dex3_pinch,
                        right_dex3_pinch,
                    )

                with dex3_direct_q_target_array.get_lock():
                    dex3_direct_q_target_array[:] = dex3_q_target
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control (keyboard/IPC use on_press; controller uses edge + long-press mapping)
            if controller_shortcut_state is not None:
                controller_shortcut_state.update(
                    tele_data,
                    allow_record_actions=True,
                    long_press_s=args.controller_right_b_long_press_s,
                )

            if args.input_mode == "controller" and args.motion:
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                else:
                    loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                      -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                      -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            if STOP:
                break

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            sol_q = np.asarray(sol_q, dtype=float)
            sol_tauff = np.asarray(sol_tauff, dtype=float)
            if sol_q.shape != current_lr_arm_q.shape or sol_tauff.shape != current_lr_arm_q.shape:
                raise SafetyCheckError(
                    f"IK output shape mismatch, aborting for safety: "
                    f"current={current_lr_arm_q.shape}, q={sol_q.shape}, tau={sol_tauff.shape}"
                )
            if not np.all(np.isfinite(sol_q)) or not np.all(np.isfinite(sol_tauff)):
                raise SafetyCheckError("IK output contains NaN/Inf, aborting for safety.")

            arm_cmd_q = sol_q
            arm_cmd_tauff = sol_tauff
            if record_start_smooth_q is not None:
                elapsed = time.time() - record_start_smooth_started_at
                speed_alpha = _smoothstep(elapsed / RECORD_START_ARM_SMOOTH_TIME)
                max_step = RECORD_START_ARM_MAX_SPEED * speed_alpha / args.frequency
                delta_q = sol_q - record_start_smooth_q
                max_delta_q = float(np.max(np.abs(delta_q))) if delta_q.size else 0.0
                if max_delta_q > max_step:
                    arm_cmd_q = record_start_smooth_q + np.clip(delta_q, -max_step, max_step)
                    record_start_smooth_q = arm_cmd_q.copy()
                else:
                    arm_cmd_q = sol_q
                    record_start_smooth_q = arm_cmd_q.copy()

                if record_start_smooth_tauff is not None:
                    tau_alpha = _smoothstep(elapsed / RECORD_START_ARM_SMOOTH_TIME)
                    arm_cmd_tauff = record_start_smooth_tauff + (sol_tauff - record_start_smooth_tauff) * tau_alpha

                if record_debug_file is not None:
                    _record_debug(
                        record_debug_file,
                        "record_smooth_frame",
                        elapsed=elapsed,
                        duration=RECORD_START_ARM_SMOOTH_TIME,
                        speed_alpha=speed_alpha,
                        max_step=max_step,
                        max_delta_q=max_delta_q,
                        current_dq_max_abs=_max_abs(current_lr_arm_dq),
                        current_to_sol_q_max_abs=_max_abs(sol_q - current_lr_arm_q),
                        current_to_cmd_q_max_abs=_max_abs(arm_cmd_q - current_lr_arm_q),
                        cmd_to_sol_q_max_abs=_max_abs(sol_q - arm_cmd_q),
                        sol_tauff_max_abs=_max_abs(sol_tauff),
                        cmd_tauff_max_abs=_max_abs(arm_cmd_tauff),
                        loop_elapsed_before_ctrl=time.time() - start_time,
                    )

                if elapsed >= RECORD_START_ARM_SMOOTH_TIME and max_delta_q <= max_step:
                    record_start_smooth_q = None
                    record_start_smooth_tauff = None
                    logger_mp.info("[Record Guard] arm target smoothing finished.")

            arm_ctrl.ctrl_dual_arm(arm_cmd_q, arm_cmd_tauff)
            last_arm_cmd_q = arm_cmd_q.copy()
            last_arm_cmd_tauff = arm_cmd_tauff.copy()

            # record mode: run filesystem work after publishing this frame's arm target.
            if pending_record_command:
                if pending_record_start:
                    if record_debug_file is not None:
                        _record_debug(
                            record_debug_file,
                            "create_episode_begin",
                            current_q_max_abs=_max_abs(current_lr_arm_q),
                            current_dq_max_abs=_max_abs(current_lr_arm_dq),
                            cmd_q_max_abs=_max_abs(arm_cmd_q),
                            cmd_tauff_max_abs=_max_abs(arm_cmd_tauff),
                        )
                    record_create_done = threading.Event()
                    record_create_result = {}
                    record_create_thread = threading.Thread(
                        target=_create_episode_worker,
                        args=(recorder, record_create_done, record_create_result),
                        daemon=True,
                    )
                    record_create_thread.start()
                    logger_mp.info("[Record Guard] create_episode started in background thread.")
                if pending_record_stop:
                    RECORD_RUNNING = False
                    record_stop_after_create = False
                    _record_debug(record_debug_file, "save_episode_begin")
                    recorder.save_episode()
                    if record_debug_file is not None:
                        record_debug_frames_remaining = int(max(1.0, RECORD_DEBUG_POST_RECORD_SECONDS) * args.frequency)
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
                if pending_record_transfer:
                    count = recorder.transfer_pending()
                    if count > 0:
                        RECORD_TRANSFER_RUNNING = True
                    _record_debug(record_debug_file, "g1_transfer_requested", transfer_count=count)

            if record_debug_file is not None and record_debug_frames_remaining > 0:
                _record_debug(
                    record_debug_file,
                    "post_record_frame",
                    record_running=RECORD_RUNNING,
                    current_q_max_abs=_max_abs(current_lr_arm_q),
                    current_dq_max_abs=_max_abs(current_lr_arm_dq),
                    cmd_q_max_abs=_max_abs(arm_cmd_q),
                    cmd_tauff_max_abs=_max_abs(arm_cmd_tauff),
                    current_to_cmd_q_max_abs=_max_abs(arm_cmd_q - current_lr_arm_q),
                    loop_elapsed_after_record_ops=time.time() - start_time,
                )
                record_debug_frames_remaining -= 1

            # record data
            if args.record:
                record_create_pending = (
                    record_create_thread is not None
                    and record_create_done is not None
                    and not record_create_done.is_set()
                )
                READY = recorder.is_ready() and not record_create_pending and not record_start_after_ready and not record_transfer_after_ready # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if (args.ee == "dex3" or args.ee == "dex3_true") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif args.ee == "dex3" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_dex3_state = dual_hand_state_array[:7]
                        right_dex3_state = dual_hand_state_array[-7:]
                    left_dex3_grip, right_dex3_grip = getattr(tv_wrapper, "_dex3_controller_grip", (0.0, 0.0))
                    left_ee_state = [_dex3_q_to_grip(left_dex3_state, DEX3_LEFT_OPEN_Q, DEX3_LEFT_CLOSED_Q)]
                    right_ee_state = [_dex3_q_to_grip(right_dex3_state, DEX3_RIGHT_OPEN_Q, DEX3_RIGHT_CLOSED_Q)]
                    left_hand_action = [left_dex3_grip]
                    right_hand_action = [right_dex3_grip]
                    current_body_state = arm_ctrl.get_current_motor_q().tolist()
                    current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                           -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                           -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif args.ee == "dex3_true" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                    current_body_state = arm_ctrl.get_current_motor_q().tolist()
                    current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                           -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                           -tele_data.right_ctrl_thumbstickValue[0] * 0.3]

                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = arm_cmd_q[:7]
                right_arm_action = arm_cmd_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    record_item_ready = True
                    if not use_g1_record:
                        if camera_config['head_camera']['binocular']:
                            if head_img is not None:
                                colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                                colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                            else:
                                logger_mp.warning("Head image is None!")
                            if camera_config['left_wrist_camera']['enable_zmq']:
                                if left_wrist_img is not None:
                                    colors[f"color_{2}"] = left_wrist_img.bgr
                                else:
                                    logger_mp.warning("Left wrist image is None!")
                            if camera_config['right_wrist_camera']['enable_zmq']:
                                if right_wrist_img is not None:
                                    colors[f"color_{3}"] = right_wrist_img.bgr
                                else:
                                    logger_mp.warning("Right wrist image is None!")
                        else:
                            if head_img is not None:
                                colors[f"color_{0}"] = head_img.bgr
                            else:
                                logger_mp.warning("Head image is None!")
                            if camera_config['left_wrist_camera']['enable_zmq']:
                                if left_wrist_img is not None:
                                    colors[f"color_{1}"] = left_wrist_img.bgr
                                else:
                                    logger_mp.warning("Left wrist image is None!")
                            if camera_config['right_wrist_camera']['enable_zmq']:
                                if right_wrist_img is not None:
                                    colors[f"color_{2}"] = right_wrist_img.bgr
                                else:
                                    logger_mp.warning("Right wrist image is None!")
                        if depth_zmq_enabled:
                            depth_array = getattr(depth_img, "depth", None) if depth_img is not None else None
                            if depth_array is None and depth_img is not None:
                                depth_array = getattr(depth_img, "bgr", None)
                            if depth_array is not None:
                                depths["depth_0"] = depth_array
                            else:
                                record_item_ready = False
                                now_for_log = time.monotonic()
                                if now_for_log - last_depth_skip_log_t > 2.0:
                                    logger_mp.warning(
                                        "[Image Record] Skipping record item because no depth frame is within sync window."
                                    )
                                    last_depth_skip_log_t = now_for_log
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if not record_item_ready:
                        pass
                    elif args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            sim_state=sim_state,
                        )
                    else:
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                        )

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            if record_debug_file is not None and args.record and (
                record_start_smooth_q is not None
                or record_debug_frames_remaining > 0
                or time_elapsed > (1.5 / args.frequency)
            ):
                _record_debug(
                    record_debug_file,
                    "loop_timing",
                    record_running=RECORD_RUNNING,
                    loop_elapsed=time_elapsed,
                    sleep_time=sleep_time,
                    current_q_max_abs=_max_abs(current_lr_arm_q),
                    current_dq_max_abs=_max_abs(current_lr_arm_dq),
                    cmd_q_max_abs=_max_abs(arm_cmd_q),
                    cmd_tauff_max_abs=_max_abs(arm_cmd_tauff),
                )
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except StartupGuardError as e:
        logger_mp.error(str(e))
    except SafetyCheckError as e:
        logger_mp.error(str(e))
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if args.record and 'recorder' in locals():
                active_on_recorder = False
                has_active_episode = getattr(recorder, "has_active_episode", None)
                if callable(has_active_episode):
                    active_on_recorder = bool(has_active_episode())
                if RECORD_RUNNING or active_on_recorder:
                    logger_mp.info("[Record Guard] stopping active episode before shutdown motion.")
                    RECORD_RUNNING = False
                    record_stop_after_create = False
                    _record_debug(record_debug_file, "shutdown_save_episode_begin")
                    recorder.save_episode()
        except Exception as e:
            logger_mp.error(f"Failed to stop active recording before shutdown motion: {e}")

        try:
            _open_end_effectors_for_shutdown(
                args,
                args.frequency,
                dex3_direct_q_target_array=locals().get('dex3_direct_q_target_array'),
                left_gripper_value=locals().get('left_gripper_value'),
                right_gripper_value=locals().get('right_gripper_value'),
            )
        except Exception as e:
            logger_mp.error(f"Failed to open end effectors during shutdown: {e}")

        try:
            if 'arm_ctrl' in locals():
                _ease_arm_to_home(arm_ctrl, args.shutdown_transition_time, args.frequency)
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join(timeout=1.0)
                if listen_keyboard_thread.is_alive():
                    logger_mp.warning("[Shutdown Guard] keyboard listener did not stop within 1s; continuing shutdown.")
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")

        try:
            if args.record and 'recorder' in locals():
                if (
                    'record_create_thread' in locals()
                    and record_create_thread is not None
                    and record_create_thread.is_alive()
                ):
                    logger_mp.warning("[Record Guard] waiting for background create_episode before closing recorder.")
                    record_create_thread.join(timeout=5.0)
                    if record_create_thread.is_alive():
                        logger_mp.warning("[Record Guard] create_episode thread is still alive; recorder.close will continue cleanup.")
                logger_mp.info("[Shutdown Guard] robot home command sent; pulling any remaining G1 episode data before exit.")
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if record_debug_file is not None:
                _record_debug(record_debug_file, "program_exit")
                record_debug_file.close()
        except Exception as e:
            logger_mp.error(f"Failed to close record debug log: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)
