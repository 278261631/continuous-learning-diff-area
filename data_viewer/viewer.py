import csv
import glob
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
        self.fig.clear()
        axes = self.fig.subplots(1, 3, sharex=True, sharey=True)

        extent = [cx - half, cx + half, cy - half, cy + half]
        titles = [
            "Template\n" + os.path.basename(rd.template_path),
            "Target (.02rp.fit)\n" + os.path.basename(rd.target_path),
            "Target - Template",
        ]
        for ax, img, lim, title in zip(axes, [t_img, g_img, diff], [t_lim, g_lim, None], titles):
            if title == "Target - Template":
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
        self.fig.suptitle(info, fontsize=11)
        self.canvas.draw()

    # ---------------------------------------------------------------- export
    def export_current(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.all_entries):
            self.statusBar().showMessage("Select a candidate row first.", 3000)
            return
        rd = self.all_entries[row]["run"]
        entries = [e for e in self.all_entries if e["run"] is rd]

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "output",
            "output_" + time.strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(out_dir, exist_ok=True)

        self._switch_fits(rd)
        half = self.half_spin.value()
        lo = self.vmin_spin.value()
        hi = self.vmax_spin.value()
        shared = self.share_scale.isChecked()

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
            for tag, img, lim in (("template", t_img, t_lim), ("target", g_img, g_lim)):
                fig = Figure(figsize=(6, 6), frameon=False)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.set_axis_off()
                ax.imshow(img, origin="lower", cmap="gray", vmin=lim[0], vmax=lim[1])
                fig.savefig(os.path.join(out_dir, f"{base}_{tag}.png"), dpi=150,
                            bbox_inches="tight", pad_inches=0)
                fig.clear()

        self.statusBar().showMessage(
            f"Exported {len(entries) * 2} images (template+target) for run '{rd.label}' -> {out_dir}"
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
