# import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import hello_imgui, imgui
from scipy.spatial import KDTree


def xy_random_normal(count: int, sigma_x: float, sigma_y: float, seed: int):
    rng = np.random.default_rng(seed)
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
            0x000000FF | (c << 24) | (c << 8),
        )


class FirefliesVisualizer:
    def __init__(self):
        self.canvas = np.zeros((1024, 1024))
        self.count = 0
        self.x = None
        self.y = None
        self.c = None
        self.freq = None
        self.phase = None
        self.sigma_x = 1
        self.sigma_y = 1
        self.mean_speed = 0.1
        self.t = 0
        self.neighbour_tree = None
        self.data = None
        self.freq_dampening = 0.9
        self.freq_step = 0.1
        self.phase_dampening = 0.9
        self.phase_step = 0.1
        self.radius = 1
        self.seed = 42

    def frame(self):
        if(hello_imgui.get_runner_params().app_shall_exit == True):
            return
        imgui.set_next_window_pos(imgui.ImVec2(0.0, 20.0))
        w, h = imgui.get_io().display_size
        imgui.set_next_window_size(imgui.ImVec2(w*0.66, h*0.9))
        imgui.begin("Fireflies")
        if not self.x is None and not self.y is None:
            self.c = (128 + 128*np.cos(self.freq*self.t + self.phase)).astype(np.uint8)
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                if np.random.random() > 0.5:
                    continue
                neighbours = self.neighbour_tree.query_ball_point((x, y), self.radius, 2)
                self.phase[i] = self.phase[i]*self.phase_dampening + self.phase_step*np.mean(self.phase[neighbours])
                self.freq[i] = self.freq[i]*self.freq_dampening + self.freq_step*np.mean(self.freq[neighbours])
            draw_fireflies(self.x, self.y, self.c)
            draw_list = imgui.get_window_draw_list()
            window_size = imgui.get_window_size()
            cursor = imgui.get_cursor_screen_pos()
            draw_list.add_circle(
                imgui.ImVec2(
                    cursor.x + window_size.x / 2, cursor.y + window_size.y / 2
                ),
                self.radius,
                0x000000FF | (255 << 24))
            
        imgui.end()
        imgui.set_next_window_pos(imgui.ImVec2(w*0.66, 0))
        imgui.set_window_size(imgui.ImVec2(0.33*w, 0.9*h))
        imgui.begin("Settings")
        reeval = False
        r1, self.count = imgui.slider_int("Count", self.count, 1, 5000)
        r2, self.sigma_x = imgui.slider_int("Sigma X", self.sigma_x, 1, 500)
        r3, self.sigma_y = imgui.slider_int("Sigma Y", self.sigma_y, 1, 500)
        r4, self.mean_speed = imgui.slider_float("Frequency", self.mean_speed, 0.001, 1)
        r5, self.freq_dampening = imgui.slider_float("Frequency Dampening", self.freq_dampening, 0.001, 1)
        if r5:
            self.freq_step = 1 - self.freq_dampening
        r7, _ = imgui.slider_float("Frequency Step", self.freq_step, 0.001, 1)
        r8, self.phase_dampening = imgui.slider_float("Phase Dampening", self.phase_dampening, 0.001, 1)
        if r8:
            self.phase_step = 1 - self.phase_dampening
        r6, _ = imgui.slider_float("Phase Step", self.phase_step, 0.001, 1)
        r9, self.radius = imgui.slider_float("Radius", self.radius, 0.001, 500)
        r10, self.seed = imgui.slider_int("Random Seed", self.seed, 1, 1024)
        reeval = any([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10])
        imgui.end()
        self.t += 1
        if reeval:
            self.x, self.y = xy_random_normal(self.count, self.sigma_x, self.sigma_y, self.seed)
            self.data = np.stack((self.x, self.y), axis=1)
            self.neighbour_tree = KDTree(data=self.data, leafsize=25)
            self.c = np.zeros_like(self.x).astype(np.uint8)
            rng = np.random.default_rng(self.seed)
            self.freq = rng.random(self.x.shape)*self.mean_speed
            self.phase = rng.random(self.x.shape)

def status():
    imgui.text("Status Bar")
if __name__ == "__main__":
    fireflies_visualizer = FirefliesVisualizer()
    callbacks = hello_imgui.RunnerCallbacks(fireflies_visualizer.frame, show_status=status)
    fps_idling = hello_imgui.FpsIdling(200, enable_idling=False)
    appwindow_params = hello_imgui.AppWindowParams("Fireflies")
    imgui_window_params = hello_imgui.ImGuiWindowParams(show_menu_bar=True, show_status_bar=True)
    runner_params = hello_imgui.RunnerParams(callbacks, appwindow_params,imgui_window_params, fps_idling=fps_idling)
    hello_imgui.run(runner_params)
