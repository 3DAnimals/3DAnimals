import logging
import random
import tarfile
from pathlib import Path

import configargparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.io as sio
from PIL import Image
from tqdm import tqdm


def download_file(url, filename):
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    response = requests.get(url, stream=True, headers=headers)
    block_size = 1024  # 1 Kibibyte

    dir_name = Path(filename).parent
    dir_name.mkdir(exist_ok=True, parents=True)
    
    with open(filename, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)


# get "maximally" different random colors:
#  ref: https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor=0.5, seed=None):
    if seed is None:
        seed = random.randint()
    r = random.Random(seed)
    return [
        (x + pastel_factor) / (1.0 + pastel_factor)
        for x in [r.uniform(0, 1.0) for i in [1, 2, 3]]
    ]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor, seed=i)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_n_colors(n, pastel_factor=0.9):
    colors = []
    for _ in range(n):
        colors.append(generate_new_color(colors, pastel_factor=pastel_factor))
    return colors


def plot_points(points, image, visible=None, correct=None):
    colors = get_n_colors(len(points), pastel_factor=0.2)
    if correct is None:
        correct = [1] * len(points)
    for i, (coord, color, visible_, correct_) in enumerate(
        zip(points, colors, visible, correct)
    ):
        if visible_ == 1:
            color = [255 * c for c in color]
            if correct_:
                image = cv2.circle(
                    np.ascontiguousarray(image),
                    tuple(coord.astype("int32")),
                    4,
                    color,
                    2,
                )
            else:
                # plot x
                image = cv2.line(
                    np.ascontiguousarray(image),
                    tuple(coord.astype("int32") - 4),
                    tuple(coord.astype("int32") + 4),
                    color,
                    2,
                )
                image = cv2.line(
                    np.ascontiguousarray(image),
                    tuple(coord.astype("int32") + np.array([-4, 4])),
                    tuple(coord.astype("int32") - np.array([-4, 4])),
                    color,
                    2,
                )
            # plot index next to point
            image = cv2.putText(
                np.ascontiguousarray(image),
                str(i),
                tuple(coord.astype("int32") + np.array([4, 4])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return image


def visualize_vertices(verts, img, cmap="hot"):
    h, w = img.shape[:2]
    verts = (verts + 1) / 2 * np.array([w, h])
    verts = np.round(verts).astype("int32")
    img = 0.5 * img + 0.5 * 255
    cmap = plt.cm.get_cmap(cmap)
    for i, v in enumerate(verts):
        x, y = v
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        img[y, x] = np.array(cmap(i / len(verts)))[:3] * 255
    return img.astype("uint8")


def arrange(images):
    rows = []
    for row in images:
        rows += [np.concatenate(row, axis=1)]
    image = np.concatenate(rows, axis=0)
    return image


def download_pascal_annotations(acsm_annotations_root):
    # download pascal annotations from dropbox (https://github.com/nileshkulkarni/acsm/blob/master/docs/setup.md)

    # The Dropbox link
    url = "https://www.dropbox.com/s/3tj037gnk4gz11t/cachedir.tar?dl=1"

    # Location to save the downloaded tar file temporarily
    tar_file = Path(acsm_annotations_root) / "cachedir.tar"

    # Download the file
    print(f"Downloading {url} to {tar_file}")
    download_file(url, tar_file)

    # Untar the file to the desired location
    print(f"Extracting {tar_file} to {acsm_annotations_root}")
    with tarfile.open(tar_file, "r") as archive:
        archive.extractall(path=acsm_annotations_root)

    print(f"File extracted to {acsm_annotations_root}")


def load_pascal_annotations(pascal_annotations_path):
    anno = sio.loadmat(
        pascal_annotations_path, struct_as_record=False, squeeze_me=True
    )["images"]
    parsed = {}
    for sample in anno:
        name = sample.rel_path[:-4] + f"_{sample.voc_rec_id}"
        box = [
            sample.bbox.x1,
            sample.bbox.y1,
            sample.bbox.x2 - sample.bbox.x1,
            sample.bbox.y2 - sample.bbox.y1,
        ]
        kp = sample.parts[:2].transpose()
        visible = sample.parts[2]
        parsed[name] = {"box": box, "kp": kp, "visible": visible}
    return parsed


def crop_keypoints(kp, name, data_dir, box_postfix):
    kp = kp.copy()

    # load our bbox
    box_path = Path(data_dir) / (name + box_postfix)
    # id, xmin, ymin, w, h, full w, full h, sharpness
    box_xmin, box_ymin, box_w, box_h = np.loadtxt(box_path)[1:5]

    # crop the original keypoints
    kp[:, 0] -= box_xmin
    kp[:, 1] -= box_ymin
    # scale keypoints to [-1, 1]
    kp[:, 0] = kp[:, 0] / box_w * 2 - 1
    kp[:, 1] = kp[:, 1] / box_h * 2 - 1

    return kp


def crop_keypoints_with_box(kp, box):
    box_xmin, box_ymin, box_w, box_h = box

    kp = kp.copy()

    # crop the original keypoints
    kp[:, 0] -= box_xmin
    kp[:, 1] -= box_ymin
    # scale keypoints to [-1, 1]
    kp[:, 0] = kp[:, 0] / box_w * 2 - 1
    kp[:, 1] = kp[:, 1] / box_h * 2 - 1

    return kp


def uncrop_keypoints_with_box(kp, box):
    """
    kp are in [-1, 1] range
    """
    box_xmin, box_ymin, box_w, box_h = box

    kp = kp.copy()

    # scale keypoints to [0, 1]
    kp[:, 0] = (kp[:, 0] + 1) / 2
    kp[:, 1] = (kp[:, 1] + 1) / 2
    # uncrop the original keypoints
    kp[:, 0] *= box_w
    kp[:, 1] *= box_h
    kp[:, 0] += box_xmin
    kp[:, 1] += box_ymin

    return kp


def load_keypoints(name, pascal_annotations, data_dir, box_postfix):
    kp = pascal_annotations[name]["kp"]
    visible = pascal_annotations[name]["visible"]
    kp = crop_keypoints(kp, name, data_dir, box_postfix)
    return kp, visible


def compute_pck(kps_err_all, visible_all, threshold):
    return (
        ((kps_err_all < threshold) * visible_all).sum(0) / visible_all.sum(0)
    ).mean()


def visualize(
    data_dir_test,
    predictions_test_dir,
    vis_dir,
    source_name,
    target_name,
    source_kp,
    target_kp,
    target_kp_pred,
    source_verts,
    target_verts,
    visible,
    kps_err,
    vert_idx,
    image_posfix,
    verts_vis_postfix,
    target_kp_pred_image,
    i,
):
    # source image
    # load image
    img_path = Path(data_dir_test) / (source_name + image_posfix)
    source_img = Image.open(img_path)

    # load verts visuals
    verts_vis_path = Path(predictions_test_dir) / (source_name + verts_vis_postfix)
    source_verts_vis = Image.open(verts_vis_path)

    # visualize keypoints
    source_kp_ = (source_kp + 1) / 2 * np.array(source_img.size)[None, :]
    source_kp_plot = plot_points(source_kp_[:, :2], source_img, visible=visible)

    source_verts_plot = visualize_vertices(source_verts, np.array(source_img))

    # target image
    # load image
    img_path = Path(data_dir_test) / (target_name + image_posfix)
    target_img = Image.open(img_path)

    # load verts visuals
    verts_vis_path = Path(predictions_test_dir) / (target_name + verts_vis_postfix)
    target_verts_vis = Image.open(verts_vis_path)

    # visualize keypoints
    target_kp_ = (target_kp + 1) / 2 * np.array(target_img.size)[None, :]
    # print(target_kp_)
    target_kp_plot = plot_points(target_kp_[:, :2], target_img, visible=visible)

    target_verts_plot = visualize_vertices(target_verts, np.array(target_img))

    target_kp_pred = (target_kp_pred + 1) / 2 * np.array(target_img.size)[None, :]
    target_kp_pred_plot = plot_points(
        target_kp_pred[:, :2], target_img, visible=visible, correct=(kps_err < 0.1)
    )

    source_kp_picked = source_verts[vert_idx]
    source_kp_picked = (source_kp_picked + 1) / 2 * np.array(source_img.size)[None, :]
    source_kp_picked_plot = plot_points(
        source_kp_picked[:, :2], source_img, visible=visible
    )

    pck1 = ((kps_err < 0.1) * visible).sum() / visible.sum()
    target_kp_pred_plot = cv2.putText(
        target_kp_pred_plot,
        f"pck@0.1: {pck1:0.4f}",
        (10, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    vis = arrange(
        [
            [source_kp_plot, target_kp_plot, target_kp_pred_plot],
            [source_kp_picked_plot, source_kp_picked_plot, source_kp_picked_plot],
            [source_verts_plot, target_verts_plot, target_verts_plot],
            [source_verts_vis, target_verts_vis, target_verts_vis],
        ]
    )

    save_path = Path(vis_dir) / "test" / f"{i:05d}-{source_name}_{target_name}.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(vis).save(save_path)


class MagicPonyResults(object):
    def __init__(
        self,
        data_dir,
        predictions_test_dir,
        vertices_postifx="_2d_projection_uv.txt",
        verts_visibility_postfix="_binary_occlusion.txt",
        box_postfix="_box.txt",
    ) -> None:
        self.data_dir = data_dir
        self.predictions_test_dir = predictions_test_dir
        self.vertices_postifx = vertices_postifx
        self.verts_visibility_postfix = verts_visibility_postfix
        self.box_postfix = box_postfix

    def load_vertices(self, name):
        # load source projected vertices
        verts_path = Path(self.predictions_test_dir) / (name + self.vertices_postifx)
        verts = np.loadtxt(verts_path)
        # load verts visibilities
        verts_visibility_path = Path(self.predictions_test_dir) / (
            name + self.verts_visibility_postfix
        )
        verts_visibility = np.loadtxt(verts_visibility_path)
        return verts, verts_visibility

    def load_box(self, name):
        # load our bbox
        box_path = Path(self.data_dir) / (name + self.box_postfix)
        # id, xmin, ymin, w, h, full w, full h, sharpness
        box_xmin, box_ymin, box_w, box_h = np.loadtxt(box_path)[1:5]
        return box_xmin, box_ymin, box_w, box_h

    def convert_keypoints(self, name, keypoints, inverse=False):
        # convert keypoints to our bounding box frame
        box = self.load_box(name)
        if inverse:
            return uncrop_keypoints_with_box(keypoints, box)
        else:
            return crop_keypoints_with_box(keypoints, box)


class Benchmark(object):
    def __init__(self, box_pad_frac=0, seed=0) -> None:
        self.box_pad_frac = box_pad_frac
        self.random = random.Random(seed)

    def sample_pair(self):
        raise NotImplementedError

    def load_keypoints(self, name):
        raise NotImplementedError

    def load_box(self, name):
        raise NotImplementedError

    def update_metric(self, name, keypoints_pred, source_visible):
        raise NotImplementedError

    def get_metric(self):
        raise NotImplementedError

    def compute_keypoints_error(self, name, keypoints_pred):
        # compute in the original image frame
        keypoints_gt, target_visible = self.load_keypoints(name)
        kps_err = np.linalg.norm(keypoints_gt - keypoints_pred, axis=-1)
        # scale error with the bounding box size
        box = self.load_box(name)
        _, _, box_w, box_h = box
        box_size = max(box_w, box_h) * (1 + 2 * self.box_pad_frac)
        kps_err = kps_err / box_size
        return kps_err, target_visible, keypoints_gt


class AcsmBenchmark(Benchmark):
    def __init__(self, acsm_annotations_root, pascal_category, **kwargs) -> None:
        super().__init__(**kwargs)
        # load pascal annotations
        # dictionary of samples indexed by image name, each sample is a dictionary with keys: box, kp, visible
        # box is in format [x_min, y_min, w, h]
        acsm_annotations_root = Path(acsm_annotations_root)
        # check if annotations are downloaded
        if not acsm_annotations_root.exists():
            download_pascal_annotations(acsm_annotations_root)
        pascal_annotations_path = (
            acsm_annotations_root
            / "cachedir"
            / "pascal"
            / "data"
            / f"{pascal_category}_val.mat"
        )
        self.pascal_annotations = load_pascal_annotations(pascal_annotations_path)

        # init metric
        self.kps_err_all = []
        self.visible_all = []

    def sample_pair(self):
        # sample source and target frames
        source_name, target_name = self.random.sample(self.pascal_annotations.keys(), 2)
        return source_name, target_name

    def load_keypoints(self, name):
        keypoints = self.pascal_annotations[name]["kp"]
        visible = self.pascal_annotations[name]["visible"]
        return keypoints, visible

    def load_box(self, name):
        box = self.pascal_annotations[name]["box"]
        return box

    def update_metric(self, name, keypoints_pred, source_visible):
        kps_err, target_visible, keypoints_gt = self.compute_keypoints_error(
            name, keypoints_pred
        )

        visible = source_visible * target_visible

        self.kps_err_all.append(kps_err)
        self.visible_all.append(visible)

        aux = {"keypoints_gt": keypoints_gt}
        return kps_err, visible, aux

    def get_metric(self):
        kps_err_all_ = np.stack(self.kps_err_all)
        visible_all_ = np.stack(self.visible_all)
        pck1 = compute_pck(kps_err_all_, visible_all_, 0.1)
        return f"pck@0.1: {pck1:0.4f}"

    def name_to_lassie_name(self, name):
        return "_".join(name.split("_")[:-1])


def transfer_keypoints(source_verts, source_verts_visibility, target_verts, source_kp):
    # tranfer keypoints to target image
    # for each keypoint in source frame, find the closest visible vertex
    source_verts[source_verts_visibility == 0] = np.inf
    dists = np.linalg.norm(source_verts[None, :, :] - source_kp[:, None, :], axis=2)
    vert_idx = np.argmin(dists, axis=1)

    # find these vertices in target frame and measrue the erorr between the target keypoint
    target_kp_pred = target_verts[vert_idx]

    aux = {"vert_idx": vert_idx}
    return target_kp_pred, aux


def main():
    # create logger
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%I:%M:%S"
    )
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    parser = configargparse.ArgumentParser(description="")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        is_config_file=True,
        help="Specify a config file path",
    )
    parser.add_argument("--data_dir_test", required=True, type=str, help="")
    parser.add_argument("--predictions_test_dir", required=True, type=str, help="")
    parser.add_argument("--acsm_annotations_root", required=True, type=str, help="")
    parser.add_argument("--pascal_category", required=True, type=str, help="")
    parser.add_argument("--vis_dir", required=False, default=None, type=str, help="")
    parser.add_argument(
        "--box_pad_frac", required=False, default=0, type=float, help=""
    )
    parser.add_argument("--n_pairs", required=False, default=10000, type=int, help="")
    parser.add_argument(
        "--vis_limit",
        required=False,
        default=10,
        type=int,
        help="Limit of visualizations to generate",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        required=False,
        default=None,
        type=str,
        help="Experiment name",
    )

    args, _ = parser.parse_known_args()

    # Replace {exp_name} in predictions_test_dir and vis_dir with exp_name argument
    if "{exp_name}" in args.predictions_test_dir:
        if args.exp_name is None:
            raise ValueError(
                "exp_name argument must be specified if {exp_name} is present in predictions_test_dir"
            )
        args.predictions_test_dir = args.predictions_test_dir.replace(
            "{exp_name}", args.exp_name
        )
    if args.vis_dir is not None and "{exp_name}" in args.vis_dir:
        if args.exp_name is None:
            raise ValueError(
                "exp_name argument must be specified if {exp_name} is present in vis_dir"
            )
        args.vis_dir = args.vis_dir.replace("{exp_name}", args.exp_name)

    verts_vis_postfix = "_2d_projection_image.png"
    image_posfix = "_rgb.png"

    results = MagicPonyResults(args.data_dir_test, args.predictions_test_dir)

    benchmark = AcsmBenchmark(
        args.acsm_annotations_root, args.pascal_category, box_pad_frac=args.box_pad_frac
    )

    results_info_path = Path(args.predictions_test_dir).parent / f"{args.exp_name}.txt"

    kps_err_all = []
    visible_all = []

    for i in tqdm(range(args.n_pairs)):
        source_name, target_name = benchmark.sample_pair()

        # load predicted vertices
        source_verts, source_verts_visibility = results.load_vertices(source_name)
        target_verts, _ = results.load_vertices(target_name)
        assert source_verts.shape == target_verts.shape

        # load source keypoints
        source_kp, source_visible = benchmark.load_keypoints(source_name)
        # convert keypoints to our bounding box frame
        source_kp = results.convert_keypoints(source_name, source_kp)

        # transfer keypoints
        target_kp_pred, transfer_aux = transfer_keypoints(
            source_verts, source_verts_visibility, target_verts, source_kp
        )

        # convert transfered keypoints to the orignal image frame
        target_kp_pred_image = results.convert_keypoints(
            target_name, target_kp_pred, inverse=True
        )

        # measrue the error between the target keypoint and transfered keypoints in image frame
        kps_err, target_visible, err_aux = benchmark.update_metric(
            target_name, target_kp_pred_image, source_visible
        )

        kps_err_all.append(kps_err)
        visible = source_visible * target_visible
        visible_all.append(visible)

        if i % 10 == 0:
            logger.info(benchmark.get_metric())
            with open(results_info_path, "w") as f:
                f.write(benchmark.get_metric())

        # visualize
        if args.vis_dir is not None and i < args.vis_limit:
            target_kp = err_aux["keypoints_gt"]
            target_kp = results.convert_keypoints(target_name, target_kp)
            vert_idx = transfer_aux["vert_idx"]
            visualize(
                args.data_dir_test,
                args.predictions_test_dir,
                args.vis_dir,
                source_name,
                target_name,
                source_kp,
                target_kp,
                target_kp_pred,
                source_verts,
                target_verts,
                visible,
                kps_err,
                vert_idx,
                image_posfix,
                verts_vis_postfix,
                target_kp_pred_image,
                i,
            )

    logger.info(benchmark.get_metric())
    with open(results_info_path, "w") as f:
        f.write(benchmark.get_metric())
    logger.info(f"Saved results to {results_info_path}")


if __name__ == "__main__":
    main()