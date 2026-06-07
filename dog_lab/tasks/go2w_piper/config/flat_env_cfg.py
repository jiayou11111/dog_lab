# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from .rough_env_cfg import Go2wPiperRoughEnvCfg, Go2wPiperRoughGraspEnvCfg


def _make_flat_terrain_generator(num_rows: int = 64, num_cols: int = 64, size: float = 2.5) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        size=(size, size),
        border_width=0.0,
        num_rows=num_rows,
        num_cols=num_cols,
        sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
    )


@configclass
class Go2wPiperFlatEnvCfg(Go2wPiperRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_flat_terrain_generator()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class Go2wPiperFlatEnvCfg_PLAY(Go2wPiperFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator = _make_flat_terrain_generator(num_rows=1, num_cols=1, size=38.0)
        self.observations.policy.enable_corruption = False
        self.events.randomize_loco_domain = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2wPiperFlatGraspEnvCfg(Go2wPiperRoughGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _make_flat_terrain_generator()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class Go2wPiperFlatGraspEnvCfg_PLAY(Go2wPiperFlatGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator = _make_flat_terrain_generator(num_rows=1, num_cols=1, size=38.0)
        self.observations.policy.enable_corruption = False
        self.events.randomize_loco_domain = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
