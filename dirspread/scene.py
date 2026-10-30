import pathlib

import numpy as np
from openpi_client import image_tools

DUMMY = [0.0] * 6 + [-1.0]
WAIT = 10
REPLAN = 5


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def photo(obs, key):
    raw = np.ascontiguousarray(obs[key][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(raw, 224, 224))


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat_wxyz(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def yaw_about(point, pos, quat, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    new_pos = point + rot @ (pos - point)
    new_quat = mat_to_quat_wxyz(rot @ quat_wxyz_to_mat(quat))
    return new_pos, new_quat


def shift_camera(env, kind):
    sim = env.env.sim
    cid = sim.model.camera_name2id("agentview")
    pos = np.array(sim.model.cam_pos[cid], dtype=np.float64)
    quat = np.array(sim.model.cam_quat[cid], dtype=np.float64)
    forward = -quat_wxyz_to_mat(quat)[:, 2]
    look = pos + forward * 1.35
    look[2] = min(look[2], 0.9)
    if kind == "yaw30":
        pos, quat = yaw_about(look, pos, quat, np.deg2rad(30))
    elif kind == "yaw-30":
        pos, quat = yaw_about(look, pos, quat, np.deg2rad(-30))
    elif kind == "yaw50":
        pos, quat = yaw_about(look, pos, quat, np.deg2rad(50))
    elif kind == "far":
        pos = look + (pos - look) * 1.45
    else:
        raise ValueError(kind)
    sim.model.cam_pos[cid] = pos
    sim.model.cam_quat[cid] = quat
    sim.forward()


def apply_delta(env, delta):
    robot = env.env.robots[0]
    idx = robot._ref_joint_pos_indexes
    q = np.array(robot.sim.data.qpos[idx], dtype=np.float64)
    q[:7] = q[:7] + np.asarray(delta, dtype=np.float64)
    robot.set_robot_joint_positions(q)


def horizontal_radius(sim, body_id):
    model = sim.model
    best = 0.0
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != int(body_id):
            continue
        pos = np.asarray(model.geom_pos[gid], dtype=np.float64)
        if int(model.geom_type[gid]) == 6:
            size = np.asarray(model.geom_size[gid], dtype=np.float64)
            corners = np.array(
                [[x, y, z] for x in (-size[0], size[0]) for y in (-size[1], size[1]) for z in (-size[2], size[2])],
                dtype=np.float64,
            )
            local = corners @ quat_wxyz_to_mat(model.geom_quat[gid]).T + pos
            reach = float(np.linalg.norm(local[:, :2], axis=1).max())
        else:
            reach = float(np.linalg.norm(pos[:2]) + model.geom_rbound[gid])
        best = max(best, reach)
    return best if best > 0 else 0.04


def bind_pair(env):
    inner = env.env
    bowl = plate = None
    for state in inner.parsed_problem["goal_state"]:
        if len(state) < 3 or str(state[0]).lower() != "on":
            continue
        if "bowl" in state[1] and "plate" in state[2]:
            bowl, plate = state[1], state[2]
            break
        if "plate" in state[1] and "bowl" in state[2]:
            plate, bowl = state[1], state[2]
            break
    if bowl is None:
        raise RuntimeError("goal has no bowl on a plate")
    sim = inner.sim
    ids = inner.obj_body_id
    env._pair = (ids[bowl], ids[plate], horizontal_radius(sim, ids[bowl]), horizontal_radius(sim, ids[plate]))


def measure(env):
    sim = env.env.sim
    bowl_id, plate_id, bowl_r, plate_r = env._pair
    bowl_pos = np.asarray(sim.data.body_xpos[bowl_id], dtype=np.float64)
    plate_pos = np.asarray(sim.data.body_xpos[plate_id], dtype=np.float64)
    up = np.asarray(sim.data.body_xmat[bowl_id], dtype=np.float64).reshape(3, 3)[:, 2]
    return np.array(
        [bowl_pos[0], bowl_pos[1], plate_pos[0], plate_pos[1], up[0], up[1], up[2], bowl_r, plate_r],
        dtype=np.float32,
    )


def open_env(suite, task_id, init_idx, camera, delta):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    init_state = suite.get_task_init_states(task_id)[init_idx]
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(init_state)
    if camera:
        shift_camera(env, camera)
        obs = env.env._get_observations()
    if delta is not None:
        apply_delta(env, delta)
        obs = env.env._get_observations()
    bind_pair(env)
    return env, obs, str(task.language)
