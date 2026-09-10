import csv
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
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

TR_MODEL_PATH = r"E:\github\simulate_astro_images\train_and_data\models_tr\best.pt"
TR_CLASS_NAMES = ("new", "brighten", "move")
TR_CLASS_COLORS = {0: "red", 1: "orange", 2: "magenta"}
TR_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("cv2") is not None
)


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


class RunData:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        with open(os.path.join(run_dir, RUN_MARKER), encoding="utf-8") as f:
            self.done = json.load(f)
        self.template_path = self.done.get("template_file")
        self.target_files = sorted(glob.glob(os.path.join(run_dir, TARGET_GLOB)))
        self.target_path = self.target_files[0] if self.target_files else None
        csv_path = os.path.join(run_dir, CSV_NAME)
        self.rows = load_csv(csv_path) if os.path.exists(csv_path) else []
        self.label = os.path.basename(run_dir)


class ViewerWindow(QMainWindow):
    def __init__(self, start_dir=None):
        super().__init__()
        self.setWindowTitle("FITS Train-Data Viewer")
        self.resize(1500, 900)

        self.run_datas = []
        self.all_entries = []
        self.current_fits_run = None
        self.template_hdul = None
        self.target_hdul = None
        self.tr_net = None
        self.tr_size = 256
        self.tr_model_path = TR_MODEL_PATH
        self.tr_torch = None

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

        btn_prev = QPushButton("< Prev")
        btn_prev.clicked.connect(lambda: self._step(-1))
        toolbar.addWidget(btn_prev)

        btn_next = QPushButton("Next >")
        btn_next.clicked.connect(lambda: self._step(1))
        toolbar.addWidget(btn_next)

        btn_export = QPushButton("Export all images")
        btn_export.clicked.connect(self.export_current)
        toolbar.addWidget(btn_export)

        self.tr_check = QCheckBox("Run TR model")
        self.tr_check.setEnabled(TR_AVAILABLE)
        if not TR_AVAILABLE:
            self.tr_check.setToolTip("torch / opencv-python not installed")
        self.tr_check.toggled.connect(lambda _: self.update_plot())
        toolbar.addWidget(self.tr_check)

        btn_tr = QPushButton("TR ckpt...")
        btn_tr.setEnabled(TR_AVAILABLE)
        btn_tr.clicked.connect(self.choose_tr_model)
        toolbar.addWidget(btn_tr)

        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["run", "rank", "x", "y", "n_frames", "median_flux", "nearest_ref_px", "ra_deg", "dec_deg"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(lambda *_: self.update_plot())
        self.table.setColumnWidth(0, 260)
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.table)

        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        splitter.addWidget(self.canvas)
        splitter.setSizes([500, 1000])

        root_layout.addWidget(splitter)

        self.setStatusBar(QStatusBar())

    # ------------------------------------------------------------------ load
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select train-data folder")
        if d:
            self.open_folder(d)

    def open_folder(self, path):
        run_dirs = collect_runs(path)
        if not run_dirs:
            self.statusBar().showMessage(
                f"No run found in {path}: need {RUN_MARKER} + {TARGET_GLOB} + {CSV_NAME}"
            )
            return

        self.run_datas = []
        self.all_entries = []
        skipped = 0
        for run_dir in run_dirs:
            rd = RunData(run_dir)
            if (
                not rd.template_path
                or not os.path.exists(rd.template_path)
                or not rd.target_path
                or not rd.rows
            ):
                skipped += 1
                continue
            self.run_datas.append(rd)
            for rec in rd.rows:
                self.all_entries.append({"run": rd, "rec": rec})

        if not self.run_datas:
            self.statusBar().showMessage(f"No usable run found under {path} (template/target/csv missing).")
            return

        self._fill_table()
        self.statusBar().showMessage(
            f"Runs: {len(self.run_datas)} (skipped {skipped}), candidates: {len(self.all_entries)}"
        )

    def _fill_table(self):
        self.table.setRowCount(len(self.all_entries))
        for i, e in enumerate(self.all_entries):
            rec = e["rec"]
            vals = [
                e["run"].label,
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
        if self.all_entries:
            self.table.setCurrentCell(0, 0)

    # ------------------------------------------------------------------ fits
    def _switch_fits(self, rd):
        if self.template_hdul is not None:
            self.template_hdul.close()
        if self.target_hdul is not None:
            self.target_hdul.close()
        self.template_hdul = fits.open(rd.template_path, memmap=True)
        self.target_hdul = fits.open(rd.target_path, memmap=True)
        self.current_fits_run = rd

    # ---------------------------------------------------------------- plotting
    def update_plot(self):
        if not self.all_entries:
            return
        row = self.table.currentRow()
        if row < 0 or row >= len(self.all_entries):
            return
        e = self.all_entries[row]
        rd = e["run"]
        rec = e["rec"]
        try:
            cx = float(rec["x"])
            cy = float(rec["y"])
        except (KeyError, ValueError):
            return

        if self.current_fits_run is not rd:
            self._switch_fits(rd)

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
        tr = None
        if self.tr_check.isChecked():
            tr = self._run_tr_panel(t_img, g_img)

        self.fig.clear()
        ncols = 4 if tr is not None else 3
        axes = list(self.fig.subplots(1, ncols, sharex=True, sharey=True).flatten())

        extent = [cx - half, cx + half, cy - half, cy + half]
        panels = [
            (t_img, t_lim, "Template\n" + os.path.basename(rd.template_path), "gray"),
            (g_img, g_lim, "Target (.02rp.fit)\n" + os.path.basename(rd.target_path), "gray"),
            (diff, None, "Target - Template", "diff"),
        ]
        for ax, (img, lim, title, kind) in zip(axes, panels):
            if kind == "diff":
                fin = diff[np.isfinite(diff)]
                q = np.percentile(np.abs(fin), 95) if fin.size else 0.0
                v = max(float(q), 1e-12)
                im = ax.imshow(img, origin="lower", extent=extent, cmap="RdBu_r", vmin=-v, vmax=v)
            else:
                im = ax.imshow(img, origin="lower", extent=extent, cmap="gray", vmin=lim[0], vmax=lim[1])
            ax.plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
            ax.set_title(title, fontsize=9)
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_xlabel("x (px)")
        axes[0].set_ylabel("y (px)")

        info = (
            f"{rd.label}  rank={rec.get('rank')}  x={cx:.2f}  y={cy:.2f}  "
            f"flux={rec.get('median_flux_norm')}  nearest_ref_px={rec.get('nearest_ref_dist_px')}  "
            f"ra={rec.get('ra_deg')}  dec={rec.get('dec_deg')}"
        )
        if tr is not None:
            self._draw_tr_panel(axes[3], g_img, g_lim, tr, extent, half, cx, cy)
            info += (
                f"\nTR: dx={tr['dx']:+.2f} dy={tr['dy']:+.2f} "
                f"droll={tr['roll']:+.2f}\u00b0  peaks={len(tr['peaks'])}"
            )
        self.fig.suptitle(info, fontsize=11)
        self.canvas.draw()

    # ------------------------------------------------------------------ tr model
    def choose_tr_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TR checkpoint", os.path.dirname(self.tr_model_path),
            "PyTorch checkpoint (*.pt);;All files (*)"
        )
        if path:
            self.tr_model_path = path
            self.tr_net = None
            self.tr_check.setChecked(True)
            self.update_plot()

    def _load_tr_model(self):
        if self.tr_net is not None:
            return True
        try:
            import torch
            from tr_model import build_model_from_state
            obj = torch.load(self.tr_model_path, map_location="cpu")
            sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            self.tr_net = build_model_from_state(sd)
            self.tr_size = int(obj.get("model_in", 256)) if isinstance(obj, dict) else 256
            self.tr_torch = torch
            return True
        except Exception as exc:
            self.tr_net = None
            self.statusBar().showMessage(f"TR model load failed: {exc}", 5000)
            return False

    @staticmethod
    def _cutout_to_u8(img):
        fin = img[np.isfinite(img)]
        if fin.size == 0:
            return np.zeros(img.shape, dtype=np.uint8)
        lo, hi = np.percentile(fin, [1.0, 99.5])
        if hi - lo <= 1e-6:
            lo, hi = float(np.nanmin(fin)), float(np.nanmax(fin))
        if hi - lo <= 1e-6:
            return np.zeros(img.shape, dtype=np.uint8)
        x = (np.nan_to_num(img, nan=lo) - lo) / (hi - lo)
        return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)

    def _run_tr_panel(self, t_img, g_img):
        if not self._load_tr_model():
            return None
        try:
            import cv2
            from tr_model import decode, heat_to_peaks, preprocess
            # FITS cutouts are displayed origin="lower"; flip to image
            # convention (row 0 = top) before feeding the model.
            a = self._cutout_to_u8(t_img)[::-1]
            b = self._cutout_to_u8(g_img)[::-1]
            size = self.tr_size
            a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA)
            pair = preprocess(a, b).unsqueeze(0)
            with self.tr_torch.no_grad():
                pose, det = self.tr_net(pair)
                dx, dy, roll = decode(pose)[0].tolist()
                if det is not None:
                    prob = self.tr_torch.sigmoid(det)
                    peaks = heat_to_peaks(prob)[0]
                    heat = prob[0].numpy()
                else:
                    peaks, heat = [], None
            return {"dx": float(dx), "dy": float(dy), "roll": float(roll),
                    "peaks": peaks, "heat": heat, "size": size}
        except Exception as exc:
            self.statusBar().showMessage(f"TR inference failed: {exc}", 5000)
            return None

    def _draw_tr_panel(self, ax, g_img, g_lim, tr, extent, half, cx, cy):
        ax.imshow(g_img, origin="lower", extent=extent, cmap="gray",
                  vmin=g_lim[0], vmax=g_lim[1])
        heat = tr["heat"]
        if heat is not None and heat.size:
            pmax = heat.max(axis=0)[::-1]
            im = ax.imshow(pmax, origin="lower", extent=extent, cmap="jet",
                           alpha=0.45, vmin=0.0, vmax=max(0.2, float(pmax.max())))
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        scale = tr["size"] / float(2 * half + 1)
        for cl, x, y, sc in tr["peaks"]:
            xa = (cx - half) + x / scale
            ya = (cy + half) - y / scale
            color = TR_CLASS_COLORS.get(cl, "white")
            ax.plot(xa, ya, "o", mfc="none", mec=color, markersize=10,
                    markeredgewidth=1.5)
            name = TR_CLASS_NAMES[cl] if 0 <= cl < len(TR_CLASS_NAMES) else str(cl)
            ax.annotate(name, (xa, ya), color=color, fontsize=7)
        ax.plot(cx, cy, "+", color="red", markersize=12, markeredgewidth=1.5)
        ax.set_title(
            f"TR model\npredict dx={tr['dx']:+.2f} dy={tr['dy']:+.2f} "
            f"droll={tr['roll']:+.2f}\u00b0", fontsize=9)

    # ---------------------------------------------------------------- export
    def export_current(self):
        if not self.all_entries:
            self.statusBar().showMessage("No candidates loaded.", 3000)
            return

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "output",
            "output_" + time.strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(out_dir, exist_ok=True)

        half = self.half_spin.value()
        lo = self.vmin_spin.value()
        hi = self.vmax_spin.value()
        shared = self.share_scale.isChecked()

        by_run = {}
        for e in self.all_entries:
            by_run.setdefault(e["run"], []).append(e)

        total = 0
        for rd, entries in by_run.items():
            if not (rd.template_path and os.path.exists(rd.template_path) and rd.target_path):
                continue
            self._switch_fits(rd)
            for e in entries:
                rec = e["rec"]
                cx, cy = float(rec["x"]), float(rec["y"])
                t_img = cutout(self.template_hdul[0].data, cx, cy, half)
                g_img = cutout(self.target_hdul[0].data, cx, cy, half)
                if shared:
                    stack = np.concatenate([t_img.ravel(), g_img.ravel()])
                    vmin, vmax = pnorm(stack, lo, hi)
                    t_lim = g_lim = (vmin, vmax)
                else:
                    t_lim = pnorm(t_img, lo, hi)
                    g_lim = pnorm(g_img, lo, hi)

                rank = rec.get("rank", "NA")
                base = f"{rd.label}_rank{rank}_x{int(round(cx))}_y{int(round(cy))}"
                npx = t_img.shape[0]
                dpi = 150
                for tag, img, lim in (("template", t_img, t_lim), ("target", g_img, g_lim)):
                    fig = Figure(figsize=(npx / dpi, npx / dpi), frameon=False)
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.set_axis_off()
                    ax.imshow(img, origin="lower", cmap="gray", vmin=lim[0], vmax=lim[1],
                              interpolation="nearest")
                    fig.savefig(os.path.join(out_dir, f"{base}_{tag}.png"), dpi=dpi,
                                bbox_inches="tight", pad_inches=0)
                    fig.clear()
                total += 2

        self.statusBar().showMessage(
            f"Exported {total} images for all {len(by_run)} runs -> {out_dir}"
        )

    # ------------------------------------------------------------- navigation
    def _step(self, delta):
        r = self.table.currentRow()
        nr = r + delta
        if 0 <= nr < self.table.rowCount():
            self.table.setCurrentCell(nr, 0)
            self.update_plot()

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
