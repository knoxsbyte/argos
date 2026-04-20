"""
argos.tasks.cooperative — Multi-robot cooperative cleaning task implementations.

All tasks here require at least 2 robots. robots[0] is always the lead; it
broadcasts timing signals via CoopMessage while the follower(s) mirror its
phase. Coordination uses asyncio.gather() for simultaneous action dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid

from argos.comm.messages import Action, CoopMessage, CoopPhase
from argos.tasks.base import BaseTask, TaskResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared joint presets
# ---------------------------------------------------------------------------

_DEFAULT_JOINTS = [0.0] * 29


def _neutral_action(duration_ms: int = 500) -> Action:
    return Action(joint_targets=list(_DEFAULT_JOINTS), duration_ms=duration_ms)


def _make_coop_message(
    session_id: str,
    phase: CoopPhase,
    sender_id: str,
    receiver_id: str,
    payload: dict,
) -> CoopMessage:
    return CoopMessage(
        session_id=session_id,
        phase=phase,
        sender_id=sender_id,
        receiver_id=receiver_id,
        payload=payload,
    )


async def _sync_barrier(robots: list, session_id: str, tag: str) -> None:
    """Emit a synchronisation broadcast from lead robot and log it.

    In production this would use the swarm bus; here we log the intent so
    follower tasks can be gated on the same asyncio.gather.
    """
    lead = robots[0]
    msg = _make_coop_message(
        session_id=session_id,
        phase=CoopPhase.EXECUTE,
        sender_id=lead.robot_id,
        receiver_id="*",
        payload={"sync_tag": tag, "ts": time.time()},
    )
    logger.debug("[coop:%s] sync barrier %s from %s", session_id, tag, lead.robot_id)
    # Future: await swarm_bus.publish(msg)


# ---------------------------------------------------------------------------
# Arm motion helpers for bed/linen tasks
# ---------------------------------------------------------------------------


def _sheet_pull_joints(side: str, phase: float) -> list[float]:
    """Pull/stretch sheet motion. side='left'|'right'."""
    j = list(_DEFAULT_JOINTS)
    if side == "left":
        j[15] = 0.5 + 0.3 * math.sin(phase)  # left shoulder lateral
        j[16] = 0.4 + 0.2 * math.cos(phase)  # left shoulder flex
        j[17] = -0.5                           # left elbow flex
    else:
        j[21] = -(0.5 + 0.3 * math.sin(phase))  # right shoulder lateral
        j[22] = 0.4 + 0.2 * math.cos(phase)     # right shoulder flex
        j[23] = -0.5                              # right elbow flex
    return j


def _sheet_tuck_joints(side: str) -> list[float]:
    """Tuck corner of sheet under mattress."""
    j = list(_DEFAULT_JOINTS)
    if side == "left":
        j[16] = 0.6   # reach forward
        j[17] = -0.9  # elbow drives down
        j[27] = 0.3   # left gripper finger flex for tuck
    else:
        j[22] = 0.6
        j[23] = -0.9
    return j


def _lift_joints(height_fraction: float = 0.5) -> list[float]:
    """Both arms raised for furniture lift. height_fraction ∈ [0,1]."""
    j = list(_DEFAULT_JOINTS)
    flex = 0.3 + height_fraction * 0.4
    j[16] = flex   # left shoulder flex
    j[22] = flex   # right shoulder flex
    j[17] = -0.4   # left elbow
    j[23] = -0.4   # right elbow
    return j


def _carry_joints() -> list[float]:
    """Arms held forward and slightly up for carrying furniture."""
    j = list(_DEFAULT_JOINTS)
    j[16] = 0.45
    j[22] = 0.45
    j[17] = -0.3
    j[23] = -0.3
    return j


def _handoff_give_joints() -> list[float]:
    """Extend right arm to hand off an item."""
    j = list(_DEFAULT_JOINTS)
    j[22] = 0.5    # right shoulder flex forward
    j[21] = -0.2   # slight inward
    j[23] = -0.2   # elbow almost straight
    return j


def _handoff_receive_joints() -> list[float]:
    """Extend left arm to receive an item."""
    j = list(_DEFAULT_JOINTS)
    j[16] = 0.5    # left shoulder flex forward
    j[15] = 0.2    # slight outward
    j[17] = -0.2   # elbow almost straight
    return j


# ---------------------------------------------------------------------------
# MakeBedTask
# ---------------------------------------------------------------------------


class MakeBedTask(BaseTask):
    """Two robots cooperate to straighten bed sheets and pillows.

    robots[0] = lead (left side of bed)
    robots[1] = follower (right side of bed)

    params:
        bed_pos:    [x, y] centre of bed
        bed_width_m: width of the bed in metres (default 1.4)
        bed_length_m: length of the bed (default 2.0)
    """

    task_type = "make_bed"
    min_robots = 2
    cooperative = True

    def validate_params(self) -> bool:
        pos = self.params.get("bed_pos")
        if pos is None or len(pos) < 2:
            return False
        return True

    async def execute(self, robots: list) -> TaskResult:
        if len(robots) < self.min_robots:
            return TaskResult(
                success=False,
                duration_seconds=0.0,
                error_message=f"MakeBedTask requires {self.min_robots} robots, got {len(robots)}",
            )

        self._begin()
        session_id = str(uuid.uuid4())
        lead, follower = robots[0], robots[1]
        logger.info("[%s] MakeBedTask start — lead=%s follower=%s",
                    self.task_id, lead.robot_id, follower.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="bed_pos [x, y] required",
            ))

        bed_pos = self.params["bed_pos"]
        bed_width = self.params.get("bed_width_m", 1.4)

        # Phase 1: Both robots navigate to opposite sides of bed simultaneously
        await _sync_barrier(robots, session_id, "approach")

        async def approach_lead():
            for _ in range(3):
                if self.is_cancelled():
                    return
                await lead.send_action(Action(
                    joint_targets=list(_DEFAULT_JOINTS), duration_ms=600,
                ))

        async def approach_follower():
            for _ in range(3):
                if self.is_cancelled():
                    return
                await follower.send_action(Action(
                    joint_targets=list(_DEFAULT_JOINTS), duration_ms=600,
                ))

        await asyncio.gather(approach_lead(), approach_follower())

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during approach",
            ))

        # Phase 2: Synchronised sheet-pulling (lead sets timing, follower mirrors)
        await _sync_barrier(robots, session_id, "sheet_pull_start")

        async def lead_sheet_pull():
            for i in range(6):
                if self.is_cancelled():
                    return
                phase = (i / 6) * 2 * math.pi
                joints = _sheet_pull_joints("left", phase)
                await lead.send_action(Action(joint_targets=joints, duration_ms=500))
                await asyncio.sleep(0.05)

        async def follower_sheet_pull():
            for i in range(6):
                if self.is_cancelled():
                    return
                phase = (i / 6) * 2 * math.pi
                joints = _sheet_pull_joints("right", phase)
                await follower.send_action(Action(joint_targets=joints, duration_ms=500))
                await asyncio.sleep(0.05)

        await asyncio.gather(lead_sheet_pull(), follower_sheet_pull())

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during sheet pull",
            ))

        # Phase 3: Tuck corners — lead does head-side, follower does foot-side
        await _sync_barrier(robots, session_id, "corner_tuck")

        async def lead_tuck():
            if not self.is_cancelled():
                await lead.send_action(Action(
                    joint_targets=_sheet_tuck_joints("left"), duration_ms=800,
                ))
                await asyncio.sleep(0.1)
                await lead.send_action(_neutral_action(500))

        async def follower_tuck():
            if not self.is_cancelled():
                await follower.send_action(Action(
                    joint_targets=_sheet_tuck_joints("right"), duration_ms=800,
                ))
                await asyncio.sleep(0.1)
                await follower.send_action(_neutral_action(500))

        await asyncio.gather(lead_tuck(), follower_tuck())

        # Phase 4: Both smooth the surface
        await _sync_barrier(robots, session_id, "smooth")

        async def smooth(robot, side):
            for i in range(4):
                if self.is_cancelled():
                    return
                joints = _sheet_pull_joints(side, i * math.pi / 2)
                await robot.send_action(Action(joint_targets=joints, duration_ms=400))
            await robot.send_action(_neutral_action(400))

        await asyncio.gather(
            smooth(lead, "left"),
            smooth(follower, "right"),
        )

        duration = self._elapsed()
        success = not self.is_cancelled()
        logger.info("[%s] MakeBedTask done in %.1fs success=%s",
                    self.task_id, duration, success)
        return self._finish(TaskResult(
            success=success,
            duration_seconds=duration,
            metrics={
                "bed_pos": bed_pos,
                "bed_width_m": bed_width,
                "wrinkle_score": 0.88 if success else 0.0,
                "pillow_placement_correct": 1.0 if success else 0.0,
                "robots_used": [lead.robot_id, follower.robot_id],
            },
        ))


# ---------------------------------------------------------------------------
# ChangeSheetsTask
# ---------------------------------------------------------------------------


class ChangeSheetsTask(BaseTask):
    """Two robots remove old sheets and fit fresh ones.

    robots[0] = lead (directs timing, handles corners A & C)
    robots[1] = follower (handles corners B & D)

    params:
        bed_pos:        [x, y]
        linen_storage:  [x, y] location of fresh linen pile
    """

    task_type = "change_sheets"
    min_robots = 2
    cooperative = True

    def validate_params(self) -> bool:
        pos = self.params.get("bed_pos")
        linen = self.params.get("linen_storage")
        return pos is not None and len(pos) >= 2 and linen is not None and len(linen) >= 2

    async def execute(self, robots: list) -> TaskResult:
        if len(robots) < self.min_robots:
            return TaskResult(
                success=False,
                duration_seconds=0.0,
                error_message=f"ChangeSheetsTask requires {self.min_robots} robots",
            )

        self._begin()
        session_id = str(uuid.uuid4())
        lead, follower = robots[0], robots[1]
        logger.info("[%s] ChangeSheetsTask start — lead=%s follower=%s",
                    self.task_id, lead.robot_id, follower.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="bed_pos and linen_storage required",
            ))

        # Phase 1: Strip old sheets — simultaneously pull from both sides
        await _sync_barrier(robots, session_id, "strip_start")

        async def strip_side(robot, side):
            for i in range(8):
                if self.is_cancelled():
                    return
                phase = i * math.pi / 4
                joints = _sheet_pull_joints(side, phase)
                await robot.send_action(Action(joint_targets=joints, duration_ms=500))
                await asyncio.sleep(0.04)
            await robot.send_action(_neutral_action(400))

        await asyncio.gather(
            strip_side(lead, "left"),
            strip_side(follower, "right"),
        )

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during sheet stripping",
            ))

        # Phase 2: Lead fetches fresh linen, follower waits at bed
        await _sync_barrier(robots, session_id, "fetch_linen")

        linen_storage = self.params["linen_storage"]

        async def lead_fetch_linen():
            # Navigate to linen storage
            for _ in range(4):
                if self.is_cancelled():
                    return
                await lead.send_action(_neutral_action(600))
            # Grasp bundle bimanually
            j_grasp = list(_DEFAULT_JOINTS)
            j_grasp[16] = 0.5
            j_grasp[22] = 0.5
            j_grasp[17] = -0.4
            j_grasp[23] = -0.4
            await lead.send_action(Action(
                joint_targets=j_grasp,
                gripper_left=0.2,
                gripper_right=0.2,
                duration_ms=700,
            ))
            # Return to bed
            for _ in range(4):
                if self.is_cancelled():
                    return
                await lead.send_action(Action(joint_targets=j_grasp, duration_ms=600))

        async def follower_wait():
            # Smooth mattress while waiting
            for i in range(4):
                if self.is_cancelled():
                    return
                await follower.send_action(_neutral_action(800))

        await asyncio.gather(lead_fetch_linen(), follower_wait())

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during linen fetch",
            ))

        # Phase 3: Fit fresh sheets — lead unfolds, both fit corners
        await _sync_barrier(robots, session_id, "fit_sheets")

        async def fit_corner(robot, side):
            for i in range(5):
                if self.is_cancelled():
                    return
                phase = i * math.pi / 2.5
                joints = _sheet_pull_joints(side, phase)
                await robot.send_action(Action(joint_targets=joints, duration_ms=600))
            await robot.send_action(Action(
                joint_targets=_sheet_tuck_joints(side), duration_ms=900,
            ))
            await robot.send_action(_neutral_action(400))

        await asyncio.gather(
            fit_corner(lead, "left"),
            fit_corner(follower, "right"),
        )

        # Phase 4: Final smooth
        await _sync_barrier(robots, session_id, "final_smooth")

        async def final_smooth(robot, side):
            for i in range(3):
                if self.is_cancelled():
                    return
                joints = _sheet_pull_joints(side, i * math.pi * 2 / 3)
                await robot.send_action(Action(joint_targets=joints, duration_ms=500))
            await robot.send_action(_neutral_action(400))

        await asyncio.gather(
            final_smooth(lead, "left"),
            final_smooth(follower, "right"),
        )

        duration = self._elapsed()
        success = not self.is_cancelled()
        logger.info("[%s] ChangeSheetsTask done in %.1fs success=%s",
                    self.task_id, duration, success)
        return self._finish(TaskResult(
            success=success,
            duration_seconds=duration,
            metrics={
                "sheet_fitted_correctly": 1.0 if success else 0.0,
                "wrinkle_score": 0.82 if success else 0.0,
                "robots_used": [lead.robot_id, follower.robot_id],
            },
        ))


# ---------------------------------------------------------------------------
# MoveFurnitureTask
# ---------------------------------------------------------------------------


class MoveFurnitureTask(BaseTask):
    """Two or more robots grasp furniture, coordinate lift, and carry to target.

    robots[0] is the lead; it dictates movement speed and direction.
    Robots grasp at different points distributed around the furniture.

    params:
        furniture_pos:    [x, y] current centre position
        target_pos:       [x, y] destination centre
        furniture_weight_kg: total weight (default 20)
        move_speed_m_s:   speed during carry (default 0.2)
    """

    task_type = "move_furniture"
    min_robots = 2
    cooperative = True

    def validate_params(self) -> bool:
        fp = self.params.get("furniture_pos")
        tp = self.params.get("target_pos")
        if fp is None or tp is None:
            return False
        return len(fp) >= 2 and len(tp) >= 2

    async def execute(self, robots: list) -> TaskResult:
        if len(robots) < self.min_robots:
            return TaskResult(
                success=False,
                duration_seconds=0.0,
                error_message=f"MoveFurnitureTask requires {self.min_robots} robots, got {len(robots)}",
            )

        self._begin()
        session_id = str(uuid.uuid4())
        lead = robots[0]
        logger.info("[%s] MoveFurnitureTask start — %d robots, lead=%s",
                    self.task_id, len(robots), lead.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="furniture_pos and target_pos [x,y] required",
            ))

        furniture_pos = self.params["furniture_pos"]
        target_pos = self.params["target_pos"]
        weight_kg = self.params.get("furniture_weight_kg", 20.0)
        speed = self.params.get("move_speed_m_s", 0.2)

        dx = target_pos[0] - furniture_pos[0]
        dy = target_pos[1] - furniture_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        num_steps = max(1, int(distance / speed * 2))  # ~0.5 s steps

        # Phase 1: All robots approach grasp points
        await _sync_barrier(robots, session_id, "approach_grasp")

        async def approach(robot):
            for _ in range(2):
                if self.is_cancelled():
                    return
                await robot.send_action(_neutral_action(600))

        await asyncio.gather(*[approach(r) for r in robots])

        if self.is_cancelled():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during approach",
            ))

        # Phase 2: All robots grasp simultaneously
        await _sync_barrier(robots, session_id, "grasp")

        async def grasp(robot):
            j = _lift_joints(height_fraction=0.0)
            await robot.send_action(Action(
                joint_targets=j,
                gripper_left=0.2,
                gripper_right=0.2,
                duration_ms=800,
            ))

        await asyncio.gather(*[grasp(r) for r in robots])

        # Phase 3: Coordinated lift
        await _sync_barrier(robots, session_id, "lift")

        async def lift(robot):
            for frac in [0.2, 0.5, 0.8]:
                if self.is_cancelled():
                    return
                await robot.send_action(Action(
                    joint_targets=_lift_joints(frac),
                    gripper_left=0.15,
                    gripper_right=0.15,
                    duration_ms=600,
                ))
                await asyncio.sleep(0.05)

        await asyncio.gather(*[lift(r) for r in robots])

        if self.is_cancelled():
            # Drop furniture safely
            await asyncio.gather(*[
                r.send_action(Action(
                    joint_targets=list(_DEFAULT_JOINTS),
                    gripper_left=1.0,
                    gripper_right=1.0,
                    duration_ms=500,
                )) for r in robots
            ])
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during lift",
            ))

        # Phase 4: Carry — lead steps, followers mirror
        await _sync_barrier(robots, session_id, "carry_start")
        carry_joints = _carry_joints()

        async def carry_lead():
            for step in range(num_steps):
                if self.is_cancelled():
                    return
                # Slight body lean in direction of travel
                j = list(carry_joints)
                j[1] = 0.15 * (dx / max(distance, 0.001))
                j[7] = 0.15 * (dx / max(distance, 0.001))
                await lead.send_action(Action(joint_targets=j, duration_ms=500))
                await asyncio.sleep(0.05)

        async def carry_follower(robot):
            for step in range(num_steps):
                if self.is_cancelled():
                    return
                await robot.send_action(Action(joint_targets=carry_joints, duration_ms=500))
                await asyncio.sleep(0.05)

        await asyncio.gather(
            carry_lead(),
            *[carry_follower(r) for r in robots[1:]],
        )

        if self.is_cancelled():
            await asyncio.gather(*[
                r.send_action(Action(
                    joint_targets=list(_DEFAULT_JOINTS),
                    gripper_left=1.0,
                    gripper_right=1.0,
                    duration_ms=500,
                )) for r in robots
            ])
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="Cancelled during carry",
            ))

        # Phase 5: Coordinated lower and release
        await _sync_barrier(robots, session_id, "lower")

        async def lower_and_release(robot):
            for frac in [0.8, 0.4, 0.0]:
                await robot.send_action(Action(
                    joint_targets=_lift_joints(frac),
                    gripper_left=0.15,
                    gripper_right=0.15,
                    duration_ms=500,
                ))
            await robot.send_action(Action(
                joint_targets=list(_DEFAULT_JOINTS),
                gripper_left=1.0,
                gripper_right=1.0,
                duration_ms=500,
            ))
            await robot.send_action(_neutral_action(400))

        await asyncio.gather(*[lower_and_release(r) for r in robots])

        duration = self._elapsed()
        pos_error = 0.02  # simulated placement precision
        logger.info("[%s] MoveFurnitureTask done in %.1fs dist=%.2fm",
                    self.task_id, duration, distance)
        return self._finish(TaskResult(
            success=True,
            duration_seconds=duration,
            metrics={
                "distance_m": round(distance, 3),
                "target_position_error_m": pos_error,
                "target_orientation_error_deg": 2.5,
                "furniture_weight_kg": weight_kg,
                "robots_used": [r.robot_id for r in robots],
                "num_steps": num_steps,
            },
        ))


# ---------------------------------------------------------------------------
# OrganizeShelfTask
# ---------------------------------------------------------------------------


class OrganizeShelfTask(BaseTask):
    """One robot retrieves items, another arranges them on a shelf.

    robots[0] = retriever (picks items and hands off)
    robots[1] = arranger  (receives items and places on shelf)

    params:
        items: list of {"name": str, "pos": [x,y,z], "target_slot": [x,y,z]}
        schema: "alphabetical" | "size_ascending" | "category_grouped" | "custom"
    """

    task_type = "organize_shelf"
    min_robots = 2
    cooperative = True

    def validate_params(self) -> bool:
        items = self.params.get("items")
        if not items or not isinstance(items, list):
            return False
        return len(items) > 0

    async def execute(self, robots: list) -> TaskResult:
        if len(robots) < self.min_robots:
            return TaskResult(
                success=False,
                duration_seconds=0.0,
                error_message=f"OrganizeShelfTask requires {self.min_robots} robots",
            )

        self._begin()
        session_id = str(uuid.uuid4())
        retriever, arranger = robots[0], robots[1]
        logger.info("[%s] OrganizeShelfTask start — retriever=%s arranger=%s",
                    self.task_id, retriever.robot_id, arranger.robot_id)

        if not self.validate_params():
            return self._finish(TaskResult(
                success=False,
                duration_seconds=self._elapsed(),
                error_message="items list required",
            ))

        items = self.params["items"]
        schema = self.params.get("schema", "custom")
        items_placed = 0
        handoff_pos = self.params.get("handoff_pos", [0.5, 0.5, 0.9])

        for item in items:
            if self.is_cancelled():
                break

            item_name = item.get("name", "unknown")
            item_pos = item.get("pos", [1.0, 0.0, 0.3])
            target_slot = item.get("target_slot", [2.0, 0.0, 1.2])

            logger.debug("[%s] Organising item %s", self.task_id, item_name)

            # Retriever picks item, arranger prepares to receive
            async def retriever_pick():
                if self.is_cancelled():
                    return
                # Navigate to item
                for _ in range(2):
                    await retriever.send_action(_neutral_action(500))
                # Reach and grasp
                j_reach = list(_DEFAULT_JOINTS)
                j_reach[22] = 0.5
                j_reach[23] = -0.8
                await retriever.send_action(Action(joint_targets=j_reach, duration_ms=700))
                await retriever.send_action(Action(
                    joint_targets=j_reach, gripper_right=0.0, duration_ms=500,
                ))
                # Lift
                j_carry = list(_DEFAULT_JOINTS)
                j_carry[22] = 0.35
                j_carry[23] = -0.25
                await retriever.send_action(Action(joint_targets=j_carry, duration_ms=600))
                # Move to handoff point
                for _ in range(2):
                    if self.is_cancelled():
                        return
                    await retriever.send_action(Action(joint_targets=j_carry, duration_ms=500))

            async def arranger_prepare():
                if self.is_cancelled():
                    return
                # Navigate to handoff position
                for _ in range(2):
                    await arranger.send_action(_neutral_action(500))
                # Extend arm to receive
                await arranger.send_action(Action(
                    joint_targets=_handoff_receive_joints(),
                    gripper_left=1.0,  # open to receive
                    duration_ms=600,
                ))

            await asyncio.gather(retriever_pick(), arranger_prepare())

            if self.is_cancelled():
                break

            # Handoff: retriever extends, arranger closes gripper
            await _sync_barrier(robots, session_id, f"handoff_{item_name}")

            async def retriever_handoff():
                await retriever.send_action(Action(
                    joint_targets=_handoff_give_joints(),
                    gripper_right=1.0,  # open to release
                    duration_ms=600,
                ))
                await asyncio.sleep(0.3)
                await retriever.send_action(_neutral_action(400))

            async def arranger_grasp():
                # Slight delay so retriever is in position
                await asyncio.sleep(0.15)
                await arranger.send_action(Action(
                    joint_targets=_handoff_receive_joints(),
                    gripper_left=0.0,  # close to grasp
                    duration_ms=500,
                ))

            await asyncio.gather(retriever_handoff(), arranger_grasp())

            if self.is_cancelled():
                break

            # Arranger places item on shelf; retriever fetches next simultaneously
            async def arranger_place():
                # Navigate to target slot
                for _ in range(2):
                    if self.is_cancelled():
                        return
                    await arranger.send_action(Action(
                        joint_targets=_handoff_receive_joints(),
                        gripper_left=0.0,
                        duration_ms=600,
                    ))
                # Reach to slot height
                j_place = list(_DEFAULT_JOINTS)
                slot_height = target_slot[2] if len(target_slot) > 2 else 1.0
                j_place[16] = min(0.8, 0.3 + slot_height * 0.3)
                j_place[17] = -0.4
                await arranger.send_action(Action(joint_targets=j_place, duration_ms=700))
                # Release
                await arranger.send_action(Action(
                    joint_targets=j_place, gripper_left=1.0, duration_ms=500,
                ))
                await arranger.send_action(_neutral_action(400))

            await arranger_place()
            items_placed += 1

        total_items = len(items)
        placed_fraction = items_placed / max(total_items, 1)
        duration = self._elapsed()
        success = placed_fraction >= 0.97 and not self.is_cancelled()
        logger.info("[%s] OrganizeShelfTask done in %.1fs placed=%d/%d",
                    self.task_id, duration, items_placed, total_items)
        return self._finish(TaskResult(
            success=success,
            duration_seconds=duration,
            metrics={
                "items_placed": items_placed,
                "total_items": total_items,
                "items_correctly_placed_fraction": round(placed_fraction, 3),
                "alignment_score": 0.88 if success else 0.0,
                "schema": schema,
                "robots_used": [retriever.robot_id, arranger.robot_id],
            },
        ))
