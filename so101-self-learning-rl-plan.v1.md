# SO-101 Self-Learning Robot Arm — Project Plan

## 1. Goal

Build a robot system that learns to manipulate objects mainly through reinforcement learning. The system should:

- perceive and label its workspace with a camera,
- accept high-level commands such as "move cup",
- reason from the goal down to actuator signals:
  *move cup → move gripper to cup → move arm → joint commands*,
- learn its own body by exploring (moving joints and observing the effect),
- reuse learned skills across different tasks.

RL is the main learning engine. Other techniques (simulation, demonstrations, foundation models) are only used to make learning faster or more efficient.

## 2. Hardware and Compute

| Component | Details |
|---|---|
| Robot | SO-101 arm (Feetech STS3215 servos, 5 DOF + gripper) |
| Camera | Intel RealSense D435 (RGB + depth), fixed, pointed at the work area |
| Training compute | Desktop with an NVIDIA GPU |
| Optional | Leader arm or teleop device for a few demonstrations |

## 3. Architecture: Hierarchical RL

Three layers. Each layer reasons about its own level of abstraction, and the lower layers are shared by all tasks.

```
 Command: "move cup"
        │
┌───────▼─────────────────────────────────────┐
│ TASK LEVEL (why)                             │
│ Goal + scene graph → sequence of skills      │
│ reach(cup) → grasp(cup) → transport(xyz)     │
│ → release()                                  │
│ v1: LLM planner   v2: learned RL policy      │
└───────┬─────────────────────────────────────┘
        │ skill + goal parameters
┌───────▼─────────────────────────────────────┐
│ SKILL LEVEL (what)                           │
│ Goal-conditioned RL policies:                │
│ reach · grasp · transport · release · push   │
│ Task-independent → transferable              │
└───────┬─────────────────────────────────────┘
        │ joint actions
┌───────▼─────────────────────────────────────┐
│ MOTOR LEVEL (how)                            │
│ World model learned from motor babbling:     │
│ state + action → next state                  │
│ Shared by all skills                         │
└───────┬─────────────────────────────────────┘
        │
   Servos (SO-101)
```

Perception runs alongside all layers and supplies the object table, gripper state and reward signals.

### Why this transfers between tasks

- **Skills are goal-conditioned.** `reach(xyz)` works for any target, not just "the cup".
- **The world model is task-independent.** Every new skill starts with a robot that already knows how its body moves.
- **The task level only composes skills.** New tasks such as "stack block" or "clear table" need new plans, not new motor learning.

## 4. Techniques That Speed Up Learning

| Technique | What it does | Where it is used |
|---|---|---|
| Hindsight Experience Replay (HER) | Relabels failed attempts as successes toward the position actually reached | All goal-conditioned skills |
| Model-based RL (TD-MPC2, with DreamerV3 as an alternative) | Trains largely inside the learned world model, so fewer real movements are needed | Skill level |
| MuJoCo pretraining | Short warm-start with domain randomization; most training happens on the real arm | Reach, and possibly grasp |
| Teleop demos (RLPD-style) | 10–20 demos seeded into the replay buffer | Grasp only |
| Foundation models | SAM2 and Grounding DINO for perception, an LLM for the initial task planner | Perception and task level |
| Self-resets | The robot resets the scene itself once basic skills work | From phase 5 |

## 5. Roadmap

### Phase 0: Foundation

- Use LeRobot drivers for the SO-101, with a ROS2 bridge as an option.
- Create **one Gym-style environment interface** shared by sim and real.
- Add a safety layer: joint limits, a Cartesian safety box, velocity and torque limits, and an emergency stop.
- Monitor servo temperature and load, and cool down automatically, because the STS3215s overheat under continuous RL.
- Calibrate the D435 (intrinsics, plus extrinsics to the robot base).
- Build a data logging pipeline for all episodes (states, actions, images, rewards).
- Set up the MuJoCo model of the SO-101, matched to the real joint limits.

**Milestone:** the same script runs a random policy in both sim and real, safely, with logging.

### Phase 1: Perception

- Take a snapshot and segment it with SAM2.
- Label the segments, either manually (click and name) or automatically with Grounding DINO.
- Build an **object table**: label, mask, 3D centroid (from depth), and approximate size.
- Track the gripper with an ArUco marker at first; replace it later with learned or segmented tracking.
- Provide a scene graph output for the task level, for example: `cup at (x,y,z), block at (...), gripper at (...)`.

**Milestone:** reliable object table updated at more than 5 Hz, with 3D positions accurate to about 1 cm.

### Phase 2: Motor Babbling and World Model

- Explore safely: random joint movements (single joints and combinations) inside the safety envelope.
- Collect 1–2 hours of real data, plus extra sim data.
- Train a dynamics model: `(joint state, action) → (next joint state, gripper position)`, later extended with a latent visual state.
- Validate by comparing predicted and actual trajectories on held-out data.
- Optionally use active exploration (curiosity: go where prediction error is high).

**Milestone:** the world model predicts gripper position over a 1-second horizon within about 1–2 cm.

### Phase 3: Reach Skill

- Goal-conditioned policy `reach(xyz)`, trained with TD-MPC2 and HER.
- Reward: negative distance from gripper to goal, with a success bonus.
- Warm-start briefly in MuJoCo, then train on the real arm.

**Milestone:** reaches arbitrary points in the workspace within about 1–2 cm, at a success rate above 90%.

### Phase 4: Grasp Skill

- Goal-conditioned on the object: `grasp(object_id)`.
- Reward: the object mask rises in depth, and gripper load increases.
- Bootstrap with 10–20 teleop demos in the replay buffer.
- Reuse the reach skill and the world model.

**Milestone:** stable grasp of 2–3 different objects at a success rate above 70%.

### Phase 5: Transport, Release and Push

- `transport(xyz)`: reach while holding an object.
- `release()`: open the gripper and check that the object is placed.
- `push(object, direction)`: non-prehensile manipulation.
- Enable self-resets so the robot puts objects back itself, allowing unattended training.

**Milestone:** full "move cup to X" works when the skills are chained by a hand-written sequence.

### Phase 6: Task Level

- **v1, LLM planner:** scene graph plus command → skill sequence with preconditions and effects. Perception checks each step, and the planner replans on failure.
- Log every plan, skill call and outcome.
- **v2, learned high-level policy:** RL over skills (options framework), with the logged LLM plans as a prior.

**Milestone:** new commands made only of existing skills work without new motor training.

## 6. Software Stack (Initial Choice)

| Area | Tool |
|---|---|
| Robot interface | LeRobot, with ROS2 as an option |
| Simulation | MuJoCo |
| RL | TD-MPC2 (main), with HER implemented in the replay buffer |
| Segmentation | SAM2 |
| Detection and labeling | Grounding DINO |
| Camera | pyrealsense2 |
| Task planner v1 | LLM (local or API) with structured output |
| Logging | LeRobot dataset format, or HDF5 |
| Experiment tracking | Weights & Biases or TensorBoard |

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Servo wear or overheating | Temperature monitoring, duty cycle limits, action smoothing |
| Manual resets slow down training | Self-resets from phase 5; a tray or bounded workspace |
| Sim-to-real gap | Keep sim light; domain randomization; most training on the real arm |
| Sparse rewards for grasping | HER, demos, dense shaped rewards from perception |
| Perception errors corrupt rewards | Validate the object table; use the ArUco marker as ground truth at first |
| Unsafe exploration | Hard safety layer independent of the policy |

## 8. Open Decisions

- [ ] Task level: when to switch from the LLM planner to the learned high-level policy, and what the criteria are
- [ ] ROS2 in the loop or pure Python with LeRobot
- [ ] Action space: joint deltas vs. end-effector deltas (Cartesian actions via learned IK)
- [ ] Visual latent state in the world model (pixels) or state-based only (object table)
- [ ] Which objects to use for the first grasp experiments
- [ ] Local or API LLM for the planner
- [ ] Control frequency (for example 10–30 Hz)

## 9. Next Steps

1. Detail Phase 0: the environment interface, safety layer and logging format.
2. Detail Phase 2: babbling strategy and world model architecture.
3. Choose the first test objects and set up the workspace.
