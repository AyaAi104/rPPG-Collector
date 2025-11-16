import os
import numpy as np
import pandas as pd

from config import data_settings
from utils.evaluate_ppg import *


class PhotoplethysmographyProcessor:
    """
    Processor for PPG files: find raw PPG CSV files, compute SQI, and save reports.
    Functionality is kept the same as the original script.
    """

    def __init__(self):
        # Channels to be used
        self.used_ch = ['Signal_Value']

        # Configuration (same as original script)
        self.sample_rate = int(70e3)  # for 'i' or 'd' mode (not used here but kept)
        self.bfi_sample_rate = 50

        # frequency windows (kept from original script)
        # frequency_windows = [[6e3, 16e3], [6e3, 26e3], [6e3, 35e3]]  # for 'i' or 'd'
        self.frequency_windows = (
            [6e3, 10.0e3],
            [10.0e3, 16.0e3],
            [16.0e3, 24.0e3],
            [24.0e3, 32.0e3],
            [32.0e3, 40.0e3],
        )

        self.show_figures = False
        self.save_figures = False
        self.data_file_name = 'HQ_proto_v4'

        # Results containers
        self.sqi_list = []
        self.all_file_sqi = []
        # Optional: save (root, SQI_final)
        self.file_sqi_records = []
        self.computing_mode = 'differential'

    # ---------- Helper functions ----------

    def remove_zeros(self, data):
        """Remove zero values from the given iterable."""
        return [value for value in data if value != 0]

    def read_ppg_file(self, file_path, file_cfg):
        """
        Read a single PPG CSV file and extract channel data.

        Returns:
            list of dicts: [{"ch": <channel_index>, "data": [values, ...]}, ...]
        """
        df = pd.read_csv(file_path)

        all_chs_data = []
        # Force column names to match the original format
        df.columns = [
            'PC_Timestamp_ms',
            'PC_DateTime',
            'Arduino_millis',
            'Signal_Value',
            'Package_Num'
        ]

        # Extract PPG column data
        for ch in self.used_ch:
            data = [abs(int(i)) for i in df[ch].dropna().values]
            data = self.remove_zeros(data)
            all_chs_data.append({"ch": 0, "data": data})
            fpath = file_path  # kept to preserve original structure (even if unused)

        return all_chs_data

    # ---------- Main processing entry ----------

    def process(self):
        """
        Main processing function: search for PPG files, compute SQI,
        and save CSV reports. This corresponds to the original top-level script.
        """
        roots = []

        # Search for folders containing the target PPG file
        for root, dirs, files in os.walk("./data/rawsignal"):
            if data_settings["ppg_input_file"] in files:
                roots.append(root)
                print(f"Find the path: {root}")

        # Process each found folder
        for root in roots:
            # Locate the PPG file
            ppg_file = os.path.join(root, data_settings["ppg_input_file"])

            if not os.path.isfile(ppg_file):
                print(f"[WARN]: path doesn't exist, pass: {ppg_file}")
                continue

            try:
                print(f"\n=== Processing File: {ppg_file} ===")
                rows = []
                ppg_datas = self.read_ppg_file(ppg_file, data_settings)

                for e in ppg_datas:
                    ch_name = e["ch"]
                    ppg_data = np.asarray(e["data"], dtype=float)

                    SQI_final, f_hr = compute_ppg_sqi(
                        ppg_data,
                        file_path=ppg_file,
                        ch=ch_name,
                        fs_in=self.bfi_sample_rate
                    )

                    rows.append({
                        "file": os.path.basename(ppg_file),
                        "ch": ch_name,
                        "SQI_final": float(SQI_final),
                        "HR_peak_Hz": float(f_hr)
                    })

                    # Save per-file SQI
                    self.all_file_sqi.append(SQI_final)
                    self.file_sqi_records.append((f"{root}:{ch_name}", float(SQI_final)))

                # Save results
                out_dir = "./data/ppg_reports"  # The first layer of saved CSV
                os.makedirs(out_dir, exist_ok=True)
                df_out = pd.DataFrame(rows)

                if not df_out.empty:
                    df_out.loc[len(df_out)] = {
                        "file": os.path.basename(ppg_file),
                        "ch": "AVG_ALL",
                        "SQI_final": df_out["SQI_final"].mean(),
                        "HR_peak_Hz": df_out["HR_peak_Hz"].mean()
                    }

                parent_folder = os.path.basename(os.path.dirname(ppg_file))
                save_dir = os.path.join(out_dir, parent_folder)
                os.makedirs(save_dir, exist_ok=True)

                out_csv = os.path.join(save_dir, "PPG_SQIs.csv")
                df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
                print(f"[OK] saved: {save_dir}")

            finally:
                i = 0  # kept for compatibility with original structure

        # The following block is kept commented as in the original code:
        # # Calculate the mean SQI across all files
        # if len(self.all_file_sqi) > 0:
        #     sqi_mean = float(np.mean(self.all_file_sqi))
        #     print("\n==================== Results: ====================")
        #     for (r, s) in self.file_sqi_records:
        #         print(f"{r}  ->  SQI={s:.4f}")
        #     print(f"\nTotal: {len(self.all_file_sqi)} Files, The mean of SQI  = {sqi_mean:.4f}")
        # else:
        #     print("\nNo SQI was successfully calculated for any file.")


if __name__ == "__main__":
    processor = PhotoplethysmographyProcessor()
    processor.process()
