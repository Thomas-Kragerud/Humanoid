

from omni.isaac.lab.terrains import (
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
)

#############################
# Rough Terrain Configuration
#############################

BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        "waves": HfWaveTerrainCfg(
            proportion=0.25,
            amplitude_range=(0.01, 0.06),
            num_waves=10,
            border_width=0.25
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.15,
            grid_height_range=(0.01, 0.04),
            platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

# TerrainGenerator procedurally generates terrains based on passed sub-terrain config
# terrain obtained either form the height fields or by using the trimesh lib
# Height field representation is more flexible but less computationally and memory efficient
# all terrain generation functions take in the argument difficulty which determines the complexity of the terrain.
# difficulty is a num [0, 1] where 1 is the hardest
# In most cases difficulty used to linearly interpolate between different terrain parameters
# ex: in pyramid stairs the step height is interpolated between specified min and max
# If curriculum is used, terrains generated based on their difficulty parameter
# difficulty varied linearly over the num of rows (x-axis) with small random value added to the difficulty, ensure columns w same sub-terrain type not exactly same
#
# difficulty = (row_id + eta)/num_rows * (upper - lower) + lower
# \eta \sim U(0, 1), and lower, upper is the range of the difficulty parameter specified in "difficulty_range"a
BLIND_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0), # width along x and y of each sub-terrain [m]
    border_width=20.0, # width of border [m]
    num_rows=4, # num rows of sub-terrains to generate
    num_cols=4,
    horizontal_scale=0.1, # discretion of the terrain along the x and y-axis [m]
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True, # cashed based on their sub-terrain config, if same sub used multiple times, only generated once
    sub_terrains={
        "waves": HfWaveTerrainCfg(
            proportion=0.25,
            amplitude_range=(0.01, 0.06),
            num_waves=10,
            border_width=0.25,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.33,
            grid_height_range=(0.01, 0.04),
            platform_width=2.0,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.34,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25
        ),
    },
    curriculum=False,
    difficulty_range=(1.0, 1.0),
)



##################################
# Hard Rough Terrain Configuration
##################################

BLIND_HARD_ROUGH_TERRAINS_CFG = BLIND_ROUGH_TERRAINS_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.02

BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG = BLIND_ROUGH_TERRAINS_PLAY_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_step = 0.02



##############################
# Stairs Terrain Configuration
##############################

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

STAIRS_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)
