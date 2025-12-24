"""
VG150 Dataset loader.

Handles loading and parsing of VG150 preprocessed files:
- VG-SGG-with-attri.h5: HDF5 file with scene graph annotations
- VG-SGG-dicts-with-attri.json: Label dictionaries
- image_data.json: Optional image metadata
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VG150Config:
    """Configuration for VG150 dataset."""

    h5_filename: str = "VG-SGG-with-attri.h5"
    dict_filename: str = "VG-SGG-dicts-with-attri.json"
    image_data_filename: str = "image_data.json"


@dataclass
class SceneGraph:
    """Scene graph for a single image."""

    image_id: int
    objects: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    attributes: list[dict[str, Any]] = field(default_factory=list)


class VG150Dataset:
    """
    Loader for VG150 preprocessed scene graph dataset.

    The VG150 dataset contains 150 object classes and 50 predicate classes,
    curated from Visual Genome for scene graph generation research.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: VG150Config | None = None,
        sample: int | None = None,
        seed: int = 42,
    ):
        """
        Initialize VG150 dataset loader.

        Args:
            data_dir: Directory containing VG150 files
            config: Dataset configuration
            sample: If set, only load first N images
            seed: Random seed (for future sampling modes)
        """
        self.data_dir = Path(data_dir)
        self.config = config or VG150Config()
        self.sample = sample
        self.seed = seed

        self._h5_file: h5py.File | None = None
        self._dicts: dict[str, Any] | None = None
        self._image_data: list[dict[str, Any]] | None = None

        # Label mappings
        self.idx_to_label: dict[int, str] = {}
        self.idx_to_predicate: dict[int, str] = {}
        self.idx_to_attribute: dict[int, str] = {}

    @property
    def h5_path(self) -> Path:
        return self.data_dir / self.config.h5_filename

    @property
    def dict_path(self) -> Path:
        return self.data_dir / self.config.dict_filename

    @property
    def image_data_path(self) -> Path:
        return self.data_dir / self.config.image_data_filename

    def validate_files(self) -> tuple[bool, list[str]]:
        """
        Validate that required files exist.

        Returns:
            Tuple of (all_valid, list of missing files)
        """
        missing = []
        if not self.h5_path.exists():
            missing.append(str(self.h5_path))
        if not self.dict_path.exists():
            missing.append(str(self.dict_path))
        return len(missing) == 0, missing

    def load(self) -> None:
        """Load dataset files into memory."""
        valid, missing = self.validate_files()
        if not valid:
            raise FileNotFoundError(
                f"Missing required VG150 files: {missing}"
            )

        # Load dictionaries
        with open(self.dict_path, "r") as f:
            self._dicts = json.load(f)

        # Build label mappings (1-indexed in VG150)
        if "idx_to_label" in self._dicts:
            self.idx_to_label = {
                int(k): v for k, v in self._dicts["idx_to_label"].items()
            }
        if "idx_to_predicate" in self._dicts:
            self.idx_to_predicate = {
                int(k): v for k, v in self._dicts["idx_to_predicate"].items()
            }
        if "idx_to_attribute" in self._dicts:
            self.idx_to_attribute = {
                int(k): v for k, v in self._dicts["idx_to_attribute"].items()
            }

        # Open H5 file (keep open for lazy loading)
        self._h5_file = h5py.File(self.h5_path, "r")

        # Load image data if available
        if self.image_data_path.exists():
            with open(self.image_data_path, "r") as f:
                self._image_data = json.load(f)

        logger.info(
            f"Loaded VG150: {len(self.idx_to_label)} objects, "
            f"{len(self.idx_to_predicate)} predicates, "
            f"{len(self.idx_to_attribute)} attributes"
        )

    def close(self) -> None:
        """Close open file handles."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __enter__(self) -> "VG150Dataset":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_num_images(self) -> int:
        """Get total number of images in dataset."""
        if self._h5_file is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # The split array indicates train/test split for each image
        total = len(self._h5_file["split"])
        if self.sample is not None:
            return min(self.sample, total)
        return total

    def get_image_ids(self) -> list[int]:
        """Get list of image IDs."""
        if self._h5_file is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Image IDs are stored in img_to_first_box or we use indices
        if "img_to_first_box" in self._h5_file:
            n_images = len(self._h5_file["img_to_first_box"])
        else:
            n_images = len(self._h5_file["split"])

        if self.sample is not None:
            n_images = min(self.sample, n_images)

        return list(range(n_images))

    def get_scene_graph(self, image_idx: int) -> SceneGraph:
        """
        Extract scene graph for a single image.

        Args:
            image_idx: Index of the image (0-based)

        Returns:
            SceneGraph object with objects, relationships, and attributes
        """
        if self._h5_file is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        h5 = self._h5_file
        sg = SceneGraph(image_id=image_idx)

        # Get box range for this image
        first_box = h5["img_to_first_box"][image_idx]
        last_box = h5["img_to_last_box"][image_idx]

        if first_box == -1:  # No boxes for this image
            return sg

        # Extract objects
        labels = h5["labels"][first_box : last_box + 1]
        boxes = h5["boxes_1024"][first_box : last_box + 1]

        for local_idx, (label_idx, box) in enumerate(zip(labels, boxes)):
            label_idx = int(label_idx)
            label = self.idx_to_label.get(label_idx, f"object_{label_idx}")
            sg.objects.append(
                {
                    "idx": first_box + local_idx,
                    "local_idx": local_idx,
                    "label_idx": label_idx,
                    "label": label,
                    "box": box.tolist() if isinstance(box, np.ndarray) else list(box),
                }
            )

        # Extract attributes if available
        if "attributes" in h5:
            attrs = h5["attributes"][first_box : last_box + 1]
            for local_idx, attr_row in enumerate(attrs):
                for attr_idx in attr_row:
                    attr_idx = int(attr_idx)
                    if attr_idx > 0:  # 0 means no attribute
                        attr_label = self.idx_to_attribute.get(
                            attr_idx, f"attr_{attr_idx}"
                        )
                        sg.attributes.append(
                            {
                                "object_local_idx": local_idx,
                                "object_label": sg.objects[local_idx]["label"],
                                "attribute_idx": attr_idx,
                                "attribute": attr_label,
                            }
                        )

        # Extract relationships
        first_rel = h5["img_to_first_rel"][image_idx]
        last_rel = h5["img_to_last_rel"][image_idx]

        if first_rel != -1:
            relationships = h5["relationships"][first_rel : last_rel + 1]
            predicates = h5["predicates"][first_rel : last_rel + 1]

            for rel, pred_idx in zip(relationships, predicates):
                subj_idx, obj_idx = int(rel[0]), int(rel[1])
                pred_idx = int(pred_idx)

                # Convert to local indices
                subj_local = subj_idx - first_box
                obj_local = obj_idx - first_box

                if 0 <= subj_local < len(sg.objects) and 0 <= obj_local < len(
                    sg.objects
                ):
                    predicate = self.idx_to_predicate.get(
                        pred_idx, f"rel_{pred_idx}"
                    )
                    sg.relationships.append(
                        {
                            "subject_idx": subj_local,
                            "subject_label": sg.objects[subj_local]["label"],
                            "predicate_idx": pred_idx,
                            "predicate": predicate,
                            "object_idx": obj_local,
                            "object_label": sg.objects[obj_local]["label"],
                        }
                    )

        return sg

    def iter_scene_graphs(self) -> Iterator[SceneGraph]:
        """Iterate over all scene graphs in the dataset."""
        for image_idx in self.get_image_ids():
            yield self.get_scene_graph(image_idx)

    def get_label_stats(self) -> dict[str, int]:
        """Get statistics about the dataset."""
        return {
            "num_object_classes": len(self.idx_to_label),
            "num_predicate_classes": len(self.idx_to_predicate),
            "num_attribute_classes": len(self.idx_to_attribute),
            "num_images": self.get_num_images(),
        }
