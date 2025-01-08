import numpy as np
import random
import cv2
import math
import perlin


class TerrainGenerator:
    BOTTOM = -10.0

    def __init__(self, width=100, height=100, dx=0.1):
        self.dx = dx
        self.size = (width, height)
        self.width = int(width // self.dx)
        self.height = int(height // self.dx)
        self.terrain = np.zeros((self.height, self.width))

    def generate_base_terrain(
        self, seed=None, octaves=5, lacunarity=2, resolution=8, scale=1.0
    ):
        if seed is None:
            seed = np.random.randint(0, 100)

        unit = lacunarity ** (octaves - 1) * resolution
        shape = ((self.height // unit + 1) * unit, (self.width // unit + 1) * unit)
        self.terrain = (
            perlin.generate_fractal_noise_2d(
                shape=shape,
                res=(resolution, resolution),
                octaves=octaves,
                tileable=(False, False),
            )[0 : self.height, 0 : self.width]
            * scale
        )

    def add_global_slope(self, direction, slope):
        # Create coordinate grids scaled by self.dx
        y_indices, x_indices = np.indices((self.height, self.width))
        x_coords = x_indices * self.dx
        y_coords = y_indices * self.dx

        # Calculate the projection of each point along the slope direction
        projection = x_coords * math.cos(direction) + y_coords * math.sin(direction)
        height_add = slope * projection

        # Add the slope to the terrain
        self.terrain += height_add

    def add_box(self, x, y, width, length, height, rotation=0):
        x = int(x // self.dx)
        y = int(y // self.dx)
        width = int(width / 2 // self.dx)
        length = int(length / 2 // self.dx)

        bounding_box = (
            max(0, int(y - width * math.sin(rotation) - length * math.cos(rotation))),
            min(
                self.terrain.shape[0],
                int(y + width * math.sin(rotation) + length * math.cos(rotation)),
            ),
            max(0, int(x - width * math.cos(rotation) - length * math.sin(rotation))),
            min(
                self.terrain.shape[1],
                int(x + width * math.cos(rotation) + length * math.sin(rotation)),
            ),
        )
        bounding_box = (
            min(bounding_box[0], bounding_box[1]),
            max(bounding_box[0], bounding_box[1]),
            min(bounding_box[2], bounding_box[3]),
            max(bounding_box[2], bounding_box[3]),
        )
        bounding_box = (
            max(0, bounding_box[0]),
            min(self.terrain.shape[0], bounding_box[1]),
            max(0, bounding_box[2]),
            min(self.terrain.shape[1], bounding_box[3]),
        )
        starting_height = np.max(
            self.terrain[
                bounding_box[0] : bounding_box[1],
                bounding_box[2] : bounding_box[3],
            ]
        )
        pts = np.array(
            [
                [
                    x + width * math.cos(rotation) + length * math.sin(rotation),
                    y + width * math.sin(rotation) - length * math.cos(rotation),
                ],
                [
                    x + width * math.cos(rotation) - length * math.sin(rotation),
                    y + width * math.sin(rotation) + length * math.cos(rotation),
                ],
                [
                    x - width * math.cos(rotation) - length * math.sin(rotation),
                    y - width * math.sin(rotation) + length * math.cos(rotation),
                ],
                [
                    x - width * math.cos(rotation) + length * math.sin(rotation),
                    y - width * math.sin(rotation) - length * math.cos(rotation),
                ],
            ]
        )
        cv2.fillPoly(self.terrain, [pts.astype(np.int32)], starting_height + height)

    def add_cylinder(self, x, y, radius, height):
        x = int(x // self.dx)
        y = int(y // self.dx)
        radius = int(radius // self.dx)
        starting_height = np.max(
            self.terrain[
                max(0, y - radius) : min(self.height, y + radius),
                max(0, x - radius) : min(self.width, x + radius),
            ]
        )

        cv2.circle(self.terrain, (x, y), radius, starting_height + height, -1)

    def add_stairs(self, num_steps, step_height, step_distance, step_width, x, y):
        step_distance = int(step_distance // self.dx)
        step_width = int(step_width // self.dx)
        x = int(x // self.dx)
        y = int(y // self.dx)
        start_height = np.max(self.terrain[y, x : x + step_width])
        for i in range(num_steps):
            self.terrain[
                y + i * step_distance : y + (i + 1) * step_distance,
                x : x + step_width,
            ] = start_height + i * step_height

    def add_gap(self, x, y, width, length, rotation):
        x = int(x // self.dx)
        y = int(y // self.dx)
        width = int(width / 2 // self.dx)
        length = int(length / 2 // self.dx)

        pts = np.array(
            [
                [
                    x + width * math.cos(rotation) + length * math.sin(rotation),
                    y + width * math.sin(rotation) - length * math.cos(rotation),
                ],
                [
                    x + width * math.cos(rotation) - length * math.sin(rotation),
                    y + width * math.sin(rotation) + length * math.cos(rotation),
                ],
                [
                    x - width * math.cos(rotation) - length * math.sin(rotation),
                    y - width * math.sin(rotation) + length * math.cos(rotation),
                ],
                [
                    x - width * math.cos(rotation) + length * math.sin(rotation),
                    y - width * math.sin(rotation) - length * math.cos(rotation),
                ],
            ]
        )
        cv2.fillPoly(self.terrain, [pts.astype(np.int32)], self.BOTTOM)

    def add_stepping_stones(
        self, width, num_stones, stone_radius, stone_spacing, x, y, direction
    ):
        width = int(width // self.dx)
        x = int(x // self.dx)
        y = int(y // self.dx)
        r = int(stone_radius // self.dx)
        stone_spacing = int(stone_spacing // self.dx)

        stone_start = (
            int(x + math.sin(direction) * width / 2),
            int(y - math.cos(direction) * width / 2),
        )
        offset = np.random.uniform(-stone_spacing, stone_spacing, num_stones)
        stone_distance = width / (num_stones + 1)
        stones = [
            np.array(
                [
                    stone_start[0]
                    - math.sin(direction) * (i + 1) * stone_distance
                    + offset[i] * math.cos(direction),
                    stone_start[1]
                    + math.cos(direction) * (i + 1) * stone_distance
                    + offset[i] * math.sin(direction),
                ],
                dtype=np.int32,
            )
            for i in range(num_stones)
        ]
        stone_heights = [
            np.mean(
                self.terrain[
                    stones[i][1] - r : stones[i][1] + r,
                    stones[i][0] - r : stones[i][0] + r,
                ]
            )
            for i in range(num_stones)
        ]
        # Dig a valley
        start = np.array(
            [
                x - math.cos(direction) * self.width,
                y - math.sin(direction) * self.height,
            ]
        )
        end = np.array(
            [
                x + math.cos(direction) * self.width,
                y + math.sin(direction) * self.height,
            ]
        )
        cv2.line(
            self.terrain,
            start.astype(np.int32),
            end.astype(np.int32),
            self.BOTTOM,
            width,
        )

        # Lay stones
        for i in range(num_stones):
            cv2.circle(
                self.terrain,
                stones[i].astype(np.int32),
                r,
                stone_heights[i],
                -1,
            )

    def auto_generate(self):
        self.generate_base_terrain(scale=random.uniform(0.0, 0.5))
        # If slope is steeper than 10 degrees, do not add stairs
        if random.uniform(0, 1) > 0.5:
            self.add_global_slope(
                random.uniform(0, math.pi / 2), random.uniform(-0.5, 0.5)
            )
        else:
            self.add_stairs(
                num_steps=20,
                step_height=random.uniform(0.1, 0.3),
                step_distance=random.uniform(0.2, 0.5),
                step_width=random.uniform(1, 5),
                x=random.uniform(0, self.size[0] / 2),
                y=random.uniform(0, self.size[1] / 2),
            )
        # add random number of random features
        num_features = random.randint(0, 20)
        if random.uniform(0, 1) < 0.5:
            # decrease feature count when stepping stones are added
            num_features = 5
            # generate stepping stones
            num_stones = random.randint(0, 10)
            stone_radius = random.uniform(0.1, 0.5)
            interval = random.uniform(0, 0.3) + stone_radius * 2
            width = (num_stones + 1) * interval
            self.add_stepping_stones(
                width,
                num_stones,
                stone_radius,
                random.uniform(0.1, 1.0),
                random.uniform(self.size[0] / 4, self.size[0] * 3 / 4),
                random.uniform(self.size[1] / 4, self.size[1] * 3 / 4),
                random.uniform(0, 2 * math.pi),
            )
        # uniform distribution of features
        feature_points = np.random.uniform(0, self.size[0], (num_features, 2))
        for i in range(num_features):
            rand = random.randint(0, 2)
            if rand == 0:
                self.add_box(
                    feature_points[i][0],
                    feature_points[i][1],
                    random.uniform(0.1, 2.0),
                    random.uniform(0.1, 2.0),
                    random.uniform(0.1, 1.0),
                    random.uniform(0, 2 * math.pi),
                )
            elif rand == 1:
                self.add_cylinder(
                    feature_points[i][0],
                    feature_points[i][1],
                    random.uniform(0.1, 2.0),
                    random.uniform(0.1, 1.0),
                )
            elif rand == 2:
                self.add_gap(
                    feature_points[i][0],
                    feature_points[i][1],
                    random.uniform(0.1, 2.0),
                    random.uniform(2.0, 5.0),
                    random.uniform(0.0, math.pi),
                )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    generator = TerrainGenerator(width=10, height=10, dx=0.05)
    # generator.generate_base_terrain(scale=0.2)
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # generator.add_global_slope(0, 0)
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # generator.add_stepping_stones(3, 6, 0.2, 0.3, 7, 3, -1)
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # generator.add_stairs(
    #     num_steps=20, step_height=0.1, step_distance=0.2, step_width=4, x=2, y=3
    # )
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # generator.add_box(7, 7, 3, 1, 0.5, math.pi / 6)
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # generator.add_cylinder(2, 9, 0.5, 0.5)
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # generator.add_gap(5, 7, 0.5, 10, -math.pi / 4)
    # plt.imshow(generator.terrain)
    # plt.show()
    #
    # x = np.linspace(0, 5, generator.width)
    # y = np.linspace(0, 5, generator.height)
    # X, Y = np.meshgrid(x, y)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, generator.terrain, rcount=200, ccount=200)
    # plt.show()

    # for _ in range(10):
    #     generator.auto_generate()
    #     x = np.linspace(0, 5, generator.width)
    #     y = np.linspace(0, 5, generator.height)
    #     X, Y = np.meshgrid(x, y)
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     ax.plot_surface(X, Y, generator.terrain, rcount=200, ccount=200)
    #     plt.show()
    #     plt.imshow(generator.terrain)
    #     plt.show()

    generator.auto_generate()
    # Benchmark time
    t = time.time()
    for _ in range(100):
        generator.auto_generate()
    print(f"time: {(time.time() - t) / 100}s avg")
