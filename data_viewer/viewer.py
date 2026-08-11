import csv
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

RUN_MARKER = ".done.json"
TARGET_GLOB = "*.02rp.fit"
CSV_NAME = "variable_candidates_nonref_only_inner_border.csv"


def is_run_dir(path):
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, RUN_MARKER))
        and len(glob.glob(os.path.join(path, TARGET_GLOB))) > 0
    )


def collect_runs(root):
    runs = []
    for dirpath, dirnames, _ in os.walk(root):
        if is_run_dir(dirpath):
            runs.append(dirpath)
            dirnames[:] = []
    return runs


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def cutout(data, cx, cy, half):
    h, w = data.shape
    x0 = int(round(cx)) - half
    x1 = int(round(cx)) + half + 1
    y0 = int(round(cy)) - half
    y1 = int(round(cy)) + half + 1
    pad_l = max(0, -x0)
    pad_r = max(0, x1 - w)
    pad_t = max(0, -y0)
    pad_b = max(0, y1 - h)
    xs = max(0, x0)
    xe = min(w, x1)
    ys = max(0, y0)
    ye = min(h, y1)
    out = np.asarray(data[ys:ye, xs:xe], dtype=float)
    if out.size == 0:
        return np.full((2 * half + 1, 2 * half + 1), np.nan)
    if pad_l or pad_r or pad_t or pad_b:
        out = np.pad(out, ((pad_t, pad_b), (pad_l, pad_r)), constant_values=np.nan)
    return out


def pnorm(img, lo_pct, hi_pct):
    fin = img[np.isfinite(img)]
    if fin.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(fin, [lo_pct, hi_pct])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(fin)), float(np.nanmax(fin))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


class ViewerWindow(QMainWindow):
    def __init__(self, start_dir=None):
        super().__init__()
        self.setWindowTitle("FITS Train-Data Viewer")
        self.resize(1500, 900)

        self.done = None
        self.template_hdul = None
        self.target_hdul = None
        self.rows = []
        self.runs = []

        self._build_ui()
        self.statusBar().showMessage("Open a train-data folder to begin.")

        if start_dir:
            self.open_folder(start_dir)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        btn_open = QPushButton("Open Folder...")
        btn_open.clicked.connect(self.choose_folder)
        toolbar.addWidget(btn_open)

        toolbar.addWidget(QLabel("Run:"))
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(280)
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        toolbar.addWidget(self.run_combo)

        toolbar.addWidget(QLabel("Target fit:"))
        self.tgt_combo = QComboBox()
        self.tgt_combo.setMinimumWidth(200)
        self.tgt_combo.currentIndexChanged.connect(self._on_tgt_changed)
        toolbar.addWidget(self.tgt_combo)

        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("Half-size (px):"))
        self.half_spin = QSpinBox()
        self.half_spin.setRange(5, 1000)
        self.half_spin.setValue(40)
        self.half_spin.valueChanged.connect(lambda _: self.update_plot())
        toolbar.addWidget(self.half_spin)

        toolbar.addWidget(QLabel("vmin %:"))
        self.vmin_spin = QSpinBox()
        self.vmin_spin.setRange(0, 100)
        self.vmin_spin.setValue(1)
        self.vmin_spin.valueChanged.connect(lambda _: self.update_plot())
        toolbar.addWidget(self.vmin_spin)

        toolbar.addWidget(QLabel("vmax %:"))
        self.vmax_spin = QSpinBox()
        self.vmax_spin.setRange(0, 100)
        self.vmax_spin.setValue(99)
        self.vmax_spin.valueChanged.connect(lambda _: self.update_plot())
        toolbar.addWidget(self.vmax_spin)

        self.share_scale = QCheckBox("Shared scale")
        self.share_scale.setChecked(True)
        self.share_scale.toggled.connect(lambda _: self.update_plot())
        toolbar.addWidget(self.share_scale)

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.addItem("skip_flag = 0")
        self.filter_combo.addItem("skip_flag != 0")
        self.filter_combo.currentIndexChanged.connect(self._refilter)
        toolbar.addWidget(QLabel("Filter:"))
        toolbar.addWidget(self.filter_combo)

        btn_prev = QPushButton("< Prev")
        btn_prev.clicked.connect(lambda: self._step(-1))
        toolbar.addWidget(btn_prev)

        self.rank_edit = QLineEdit()
        self.rank_edit.setFixedWidth(70)
        self.rank_edit.returnPressed.connect(self._jump_to_rank)
        toolbar.addWidget(QLabel("Rank:"))
        toolbar.addWidget(self.rank_edit)

        btn_next = QPushButton("Next >")
        btn_next.clicked.connect(lambda: self._step(1))
        toolbar.addWidget(btn_next)

        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["rank", "x", "y", "n_frames", "median_flux", "nearest_ref_px", "ra_deg", "dec_deg"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(lambda *_: self.update_plot())
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)
        splitter.setSizes([420, 1080])

        root_layout.addWidget(splitter)

        self.setStatusBar(QStatusBar())

    # ------------------------------------------------------------------ load
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select train-data folder")
        if d:
            self.open_folder(d)

    def open_folder(self, path):
        self.runs = collect_runs(path)
        if not self.runs:
            self.statusBar().showMessage(
                f"No run found in {path}: need {RUN_MARKER} + {TARGET_GLOB} + {CSV_NAME}"
            )
            return
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for r in self.runs:
            self.run_combo.addItem(os.path.relpath(r, path) if r != path else os.path.basename(path) or path)
        self.run_combo.blockSignals(False)
        self._load_run(self.runs[0])

    def _on_run_changed(self, idx):
        if 0 <= idx < len(self.runs):
            self._load_run(self.runs[idx])

    def _load_run(self, run_dir):
        done_path = os.path.join(run_dir, RUN_MARKER)
        with open(done_path, encoding="utf-8") as f:
            self.done = json.load(f)

        self.template_path = self.done.get("template_file")
        if not self.template_path or not os.path.exists(self.template_path):
            self.statusBar().showMessage(
                f"Template not found: {self.template_path!r} (set in {done_path})"
            )
            return

        self.tgt_files = sorted(glob.glob(os.path.join(run_dir, TARGET_GLOB)))
        if not self.tgt_files:
            self.statusBar().showMessage(f"No {TARGET_GLOB} in {run_dir}")
            return

        self.tgt_combo.blockSignals(True)
        self.tgt_combo.clear()
        for t in self.tgt_files:
            self.tgt_combo.addItem(os.path.basename(t), t)
        self.tgt_combo.blockSignals(False)

        csv_path = os.path.join(run_dir, CSV_NAME)
        if not os.path.exists(csv_path):
            self.statusBar().showMessage(f"Missing {CSV_NAME} in {run_dir}")
            return
        self.all_rows = load_csv(csv_path)

        self._open_fits(self.template_path, self.tgt_files[0])
        self._refilter()
        self.statusBar().showMessage(
            f"Run: {run_dir} | template: {os.path.basename(self.template_path)} | "
            f"target: {os.path.basename(self.tgt_files[0])} | candidates: {len(self.rows)}"
        )

    def _on_tgt_changed(self):
        path = self.tgt_combo.currentData()
        if path and self.template_path:
            self._open_fits(self.template_path, path)

    def _open_fits(self, template_path, target_path):
        if self.template_hdul is not None:
            self.template_hdul.close()
        if self.target_hdul is not None:
            self.target_hdul.close()
        self.template_hdul = fits.open(template_path, memmap=True)
        self.target_hdul = fits.open(target_path, memmap=True)
        self.update_plot()

    # --------------------------------------------------------------- filtering
    def _refilter(self):
        if not getattr(self, "all_rows", None):
            return
        mode = self.filter_combo.currentIndex()
        self.rows = []
        for rec in self.all_rows:
            try:
                flag = int(float(rec.get("skip_flag", "0")))
            except (TypeError, ValueError):
                flag = 0
            if mode == 1 and flag != 0:
                continue
            if mode == 2 and flag == 0:
                continue
            self.rows.append(rec)
        self._fill_table()

    def _fill_table(self):
        self.table.setRowCount(len(self.rows))
        for i, rec in enumerate(self.rows):
            vals = [
                rec.get("rank", ""),
                rec.get("x", ""),
                rec.get("y", ""),
                rec.get("n_frames_detected", ""),
                rec.get("median_flux_norm", ""),
                rec.get("nearest_ref_dist_px", ""),
                rec.get("ra_deg", ""),
                rec.get("dec_deg", ""),
            ]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))
        if self.rows:
            self.table.setCurrentCell(0, 0)

    # ---------------------------------------------------------------- plotting
    def update_plot(self):
        if self.template_hdul is None or self.target_hdul is None:
            return
        row = self.table.currentRow()
        if row < 0 or row >= len(self.rows):
            return
        rec = self.rows[row]
        try:
            cx = float(rec["x"])
            cy = float(rec["y"])
        except (KeyError, ValueError):
            return

        half = self.half_spin.value()
        t_img = cutout(self.template_hdul[0].data, cx, cy, half)
        g_img = cutout(self.target_hdul[0].data, cx, cy, half)

        lo = self.vmin_spin.value()
        hi = self.vmax_spin.value()
        if self.share_scale.isChecked():
            stack = np.concatenate([t_img.ravel(), g_img.ravel()])
            vmin, vmax = pnorm(stack, lo, hi)
            t_lim = g_lim = (vmin, vmax)
        else:
            t_lim = pnorm(t_img, lo, hi)
            g_lim = pnorm(g_img, lo, hi)

        diff = g_img - t_img
        self.fig.clear()
        axes = self.fig.subplots(1, 3, sharex=True, sharey=True)

        extent = [cx - half, cx + half, cy - half, cy + half]
        titles = ["Template\n" + os.path.basename(self.template_path),
                  "Target (.02rp.fit)\n" + os.path.basename(self.tgt_combo.currentText()),
                  "Target - Template"]
        for ax, img, lim, title in zip(axes, [t_img, g_img, diff], [t_lim, g_lim, None], titles):
            if title.startswith("Target - Template"):
                fin = diff[np.isfinite(diff)]
                q = np.percentile(np.abs(fin), 95) if fin.size else 0.0
                v = max(float(q), 1e-12)
                im = ax.imshow(img, origin="lower", extent=extent, cmap="RdBu_r",
                               vmin=-v, vmax=v)
            else:
                im = ax.imshow(img, origin="lower", extent=extent, cmap="gray",
                               vmin=lim[0], vmax=lim[1])
            ax.plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
            ax.set_title(title, fontsize=9)
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_xlabel("x (px)")
        axes[0].set_ylabel("y (px)")

        info = (
            f"rank={rec.get('rank')}  x={cx:.2f}  y={cy:.2f}  "
            f"flux={rec.get('median_flux_norm')}  nearest_ref_px={rec.get('nearest_ref_dist_px')}  "
            f"ra={rec.get('ra_deg')}  dec={rec.get('dec_deg')}  skip={rec.get('skip_flag')}"
        )
        self.fig.suptitle(info, fontsize=11)
        self.canvas.draw()

    # ------------------------------------------------------------- navigation
    def _step(self, delta):
        r = self.table.currentRow()
        nr = r + delta
        if 0 <= nr < self.table.rowCount():
            self.table.setCurrentCell(nr, 0)
            self.update_plot()

    def _jump_to_rank(self):
        try:
            target = int(self.rank_edit.text().strip())
        except ValueError:
            return
        for i, rec in enumerate(self.rows):
            try:
                if int(float(rec.get("rank", "-1"))) == target:
                    self.table.setCurrentCell(i, 0)
                    self.update_plot()
                    return
            except ValueError:
                continue
        self.statusBar().showMessage(f"No candidate with rank {target}", 3000)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up,):
            self._step(-1)
        elif event.key() in (Qt.Key_Down,):
            self._step(1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.template_hdul is not None:
            self.template_hdul.close()
        if self.target_hdul is not None:
            self.target_hdul.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    start = sys.argv[1] if len(sys.argv) > 1 else None
    win = ViewerWindow(start_dir=start)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
