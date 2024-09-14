# import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import hello_imgui, imgui, immvision
import cv2


def xy_random_normal(count: int, sigma_x: float, sigma_y: float):
    rng = np.random.default_rng(42)
    xy = rng.normal((0.0, 0.0), (sigma_x, sigma_y), (count, 2))
    return xy[:, 0], xy[:, 1]


def xy_random_uniform(count: int, low: tuple[int, int], high: tuple[int, int]):
    rng = np.random.default_rng(42)
    xy = rng.uniform(low, high, (count, 2))
    return xy[:, 0], xy[:, 1]


def test_xy_random_normal():
    for i in range(25):
        sigma_x, sigma_y = np.random.uniform(1, 50, 2)
        count = (i + 1) * 10000
        x, y = xy_random_normal(count, sigma_x, sigma_y)
        np.testing.assert_allclose(np.mean(x), [0.0], atol=1e-1 * sigma_x)
        np.testing.assert_allclose(np.mean(y), [0.0], atol=1e-1 * sigma_y)
        np.testing.assert_allclose(np.std(x), sigma_x, atol=1e-1 * sigma_x)
        np.testing.assert_allclose(np.std(y), sigma_y, atol=1e-1 * sigma_y)


def test_xy_random_uniform():
    for i in range(25):
        count = (i + 1) * 10000
        low_x, high_x = np.random.uniform(1, 50, 2)
        if high_x < low_x:
            low_x, high_x = high_x, low_x
        low_y, high_y = np.random.uniform(1, 50, 2)
        if high_y < low_y:
            low_y, high_y = high_y, low_y

        x, y = xy_random_uniform(count, (low_x, low_y), (high_x, high_y))
        np.testing.assert_allclose(np.mean(x), (low_x + high_x) / 2, atol=0.5)
        np.testing.assert_allclose(np.mean(y), (low_y + high_y) / 2, atol=0.5)
        np.testing.assert_allclose(np.max(x), high_x, atol=0.05)
        np.testing.assert_allclose(np.min(x), low_x, atol=0.05)
        np.testing.assert_allclose(np.max(y), high_y, atol=0.05)
        np.testing.assert_allclose(np.min(y), low_y, atol=0.05)


def draw_fireflies(x, y, lightness):
    draw_list = imgui.get_window_draw_list()
    window_size = imgui.get_window_size()
    cursor = imgui.get_cursor_screen_pos()
    for x, y, c in zip(x, y, lightness):
        # transform_mat = self.image_params.zoom_pan_matrix
        draw_list.add_circle_filled(
            imgui.ImVec2(
                cursor.x + x + window_size.x / 2, cursor.y + y + window_size.y / 2
            ),
            10,
            0x000000FF | (c << 24),
        )


class FirefliesVisualizer:
    def __init__(self):
        self.canvas = np.zeros((1024, 1024))
        self.image_params = immvision.ImageParams()
        self.count = 0
        self.x = None
        self.y = None
        self.c = None
        self.freq = None
        self.sigma_x = 1
        self.sigma_y = 1
        self.mean_speed = 0.1
        self.t = 0

    def frame(self):
        imgui.begin("Fireflies")
        if not self.x is None and not self.y is None:
            self.c = (128 + 128*np.cos(self.freq*self.t)).astype(np.uint8)
            draw_fireflies(self.x, self.y, self.c)
        imgui.end()
        imgui.begin("Settings")
        reeval = False
        r1, self.count = imgui.slider_int("Count", self.count, 1, 500)
        r2, self.sigma_x = imgui.slider_int("Sigma X", self.sigma_x, 1, 500)
        r3, self.sigma_y = imgui.slider_int("Sigma Y", self.sigma_y, 1, 500)
        r4, self.mean_speed = imgui.slider_float("Frequency", self.mean_speed, 0.001, 1)
        reeval = r1 or r2 or r3 or r4
        imgui.end()
        self.t += 1
        if reeval:
            self.x, self.y = xy_random_normal(self.count, self.sigma_x, self.sigma_y)
            self.c = np.zeros_like(self.x).astype(np.uint8)
            self.freq = np.random.random(self.x.shape)*self.mean_speed


if __name__ == "__main__":
    fireflies_visualizer = FirefliesVisualizer()
    callbacks = hello_imgui.RunnerCallbacks(fireflies_visualizer.frame)
    fps_idling = hello_imgui.FpsIdling(30, enable_idling=False)
    appwindow_params = hello_imgui.AppWindowParams("Fireflies")
    runner_params = hello_imgui.RunnerParams(callbacks, appwindow_params, fps_idling=fps_idling)
    hello_imgui.run(runner_params)
