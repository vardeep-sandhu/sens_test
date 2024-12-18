import time
import click
import numpy as np
import threading

import sys
from utils import *
import open3d as o3d
import pynput.keyboard as keyboard


class VisDetection:
    def __init__(
        self, scan_files, detection_files=None, gt_annos_files=None, only_gt=None
    ):
        # init file paths
        self.only_gt = only_gt
        self.scan_files = scan_files
        self.detection_files = detection_files
        self.gt_annos_files = gt_annos_files

        if self.detection_files is not None:
            self.times = list(detection_files)

        self.current_points, self.current_colors = load_vertex(scan_files[0])
        scan_basepath = os.path.basename(self.scan_files[0])
        self.scan_name = os.path.splitext(scan_basepath)[0]

        if self.detection_files is not None:
            self.curr_detections, self.classes, self.current_gt = (
                load_dets_n_gt_per_frame(
                    self.detection_files[self.scan_name],
                    self.gt_annos_files[self.scan_name],
                )
            )

        # init visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
        self.pcd.colors = o3d.utility.Vector3dVector(self.current_colors)

        # self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        print("hello")

        # bbox = o3d.geometry.AxisAlignedBoundingBox(
        #     min_bound=(-45, -45, -5), max_bound=(45, 45, 5)
        # )
        # print("hello")

        # self.pcd = self.pcd.crop(bbox)  # set view area
        print("hello")

        self.vis.add_geometry(self.pcd)
        print("hello")

        # init keyboard controller
        key_listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        print("hello")

        key_listener.start()
        print("hello")

        # init frame index
        self.frame_idx = 0
        self.num_frames = len(self.scan_files)
        # init threading
        self.lock = threading.Lock()
        self.load_next_frame = True
        self.load_prev_frame = True
        self.worker_thread_next = threading.Thread(target=self.worker_next)
        self.worker_thread_prev = threading.Thread(target=self.worker_prev)

        self.worker_thread_next.start()
        self.worker_thread_prev.start()

    def on_press(self, key):
        try:
            if key.char == "q":
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

            if key.char == "n":
                with self.lock:
                    self.load_next_frame = True

            if key.char == "b":
                with self.lock:
                    self.load_prev_frame = True

        except AttributeError:
            print("special key {0} pressed".format(key))

    def on_release(self, key):
        try:
            if key.char == "n" or key.char == "b":
                self.current_points, self.current_colors = load_vertex(
                    self.scan_files[self.frame_idx]
                )
                scan_basepath = os.path.basename(self.scan_files[self.frame_idx])
                self.scan_name = os.path.splitext(scan_basepath)[0]

                if self.detection_files is not None:
                    self.curr_detections, self.classes, self.current_gt = (
                        load_dets_n_gt_per_frame(
                            self.detection_files[self.scan_name],
                            self.gt_annos_files[self.scan_name],
                        )
                    )

        except AttributeError:
            print("special key {0} pressed".format(key))

    def run(self):
        self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
        self.pcd.colors = o3d.utility.Vector3dVector(self.current_colors)
        if self.detection_files is not None:
            o3d_box = draw_bbx_arrows(
                self.curr_detections,
                self.classes,
                self.current_gt,
                only_gt=self.only_gt,
            )

        self.vis.clear_geometries()
        if self.detection_files is not None:
            geometry = [self.pcd] + o3d_box
        else:
            geometry = [self.pcd]

        for i in geometry:
            self.vis.add_geometry(i, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.01)

    def worker_next(self):
        while True:
            if self.load_next_frame:
                with self.lock:
                    if self.frame_idx < self.num_frames - 1:
                        self.frame_idx += 1
                        self.current_points, self.current_colors = load_vertex(
                            self.scan_files[self.frame_idx]
                        )
                        scan_basepath = os.path.basename(
                            self.scan_files[self.frame_idx]
                        )
                        self.scan_name = os.path.splitext(scan_basepath)[0]
                        if self.detection_files is not None:

                            self.curr_detections, self.classes, self.current_gt = (
                                load_dets_n_gt_per_frame(
                                    self.detection_files[self.scan_name],
                                    self.gt_annos_files[self.scan_name],
                                )
                            )
                        print("frame index:", self.frame_idx)
                    else:
                        print("Reach the end of this sequence!")

                self.load_next_frame = False

            time.sleep(0.01)

    def worker_prev(self):
        while True:
            if self.load_prev_frame:
                with self.lock:
                    if self.frame_idx > 1:
                        self.frame_idx -= 1
                        self.current_points, self.current_colors = load_vertex(
                            self.scan_files[self.frame_idx]
                        )
                        scan_basepath = os.path.basename(
                            self.scan_files[self.frame_idx]
                        )
                        self.scan_name = os.path.splitext(scan_basepath)[0]
                        if self.detection_files is not None:

                            self.curr_detections, self.classes, self.current_gt = (
                                load_dets_n_gt_per_frame(
                                    self.detection_files[self.scan_name],
                                    self.gt_annos_files[self.scan_name],
                                )
                            )
                        print("frame index:", self.frame_idx)
                    else:
                        print("At the start at this sequence!")

                self.load_prev_frame = False

            time.sleep(0.01)


@click.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--detection_path",
    "-det",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--gt_anno_path",
    "-gt",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--only_gt",
    is_flag=True,
)
def main(data, detection_path=None, gt_anno_path=None, only_gt=False):
    scan_paths = load_files(data)
    # detections = load_data_pkl(detection_path)
    # gt_annos = load_data_pkl(gt_anno_path)

    # assert len(scan_paths) == len(detections)

    visualizer = VisDetection(
        scan_paths, detection_files=None, gt_annos_files=None, only_gt=only_gt
    )

    while True:
        visualizer.run()


if __name__ == "__main__":
    path = "/home/sandhu/learning/sensmore_test/SemanticKITTI_00/velodyne"
    main()
