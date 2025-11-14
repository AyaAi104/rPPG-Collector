from config import data_settings
from utils.evaluate_ppg import *


import pandas as pd

used_ch=['Signal_Value']

def remove_zeros(data):
    return [value for value in data if value != 0]


def process_ppg_file(file_path, file):
    df = pd.read_csv(file_path)

    all_chs_data = []
    #
    df.columns = ['PC_Timestamp_ms', 'PC_DateTime', 'Arduino_millis', 'Signal_Value',
       'Package_Num']

    # 提取 PPG1 列数据并计算 PI
    for ch in used_ch:
        # data = [-abs(int(i)) for i in df['ch2'].dropna().values]
        data = [abs(int(i)) for i in df[ch].dropna().values]
        data = remove_zeros(data)
        all_chs_data.append({"ch": 0, "data": data})
        fpath = file_path
    return all_chs_data


sample_rate = int(70e3)  # 'i'或'd'模式用
bfi_sample_rate = 50
# frequency_windows = [[6e3, 16e3], [6e3, 26e3], [6e3, 35e3]]  # 对应 'i' 或 'd'
frequency_windows = (
    [6e3, 10.0e3],
    [10.0e3, 16.0e3],
    [16.0e3, 24.0e3],
    [24.0e3, 32.0e3],
    [32.0e3, 40.0e3],
)
show_figures = False
save_figures = False
data_file_name = 'HQ_proto_v4'

roots = []
for root, dirs, files in os.walk("./data/rawsignal"):
    if (data_settings["ppg_input_file"] in files):
        roots.append(root)
        print(f"Find the path: {root}")

# save results
sqi_list = []
all_file_sqi = []
file_sqi_records = []  # 可选：保存 (root, sxi_final)
computing_mode = 'differential'

for root in roots:
    # Locate the LDF file
    ppg_file = os.path.join(root, data_settings["ppg_input_file"])

    if not os.path.isfile(ppg_file):
        print(f"[WARN]: path doesn't exist, pass：{ppg_file}")
        continue

    try:
        print(f"\n=== Processing File: {ppg_file} ===")
        rows = []
        ppg_datas = process_ppg_file(ppg_file, data_settings)
        for e in ppg_datas:
            ch_name = e["ch"]
            ppg_data = np.asarray(e["data"], dtype=float)

            #
            SQI_final, f_hr = compute_ppg_sqi(
                ppg_data,
                file_path=ppg_file,
                ch=ch_name,
                fs_in=bfi_sample_rate
            )
            rows.append({
                "file": os.path.basename(ppg_file),
                "ch": ch_name,
                "SQI_final": float(SQI_final),
                "HR_peak_Hz": float(f_hr)
            })

            # compute the BFi waveforms using the computed PSDs at given frequency windows
            # SQI_final = compute_ppg_sqi(ppg_data, fs_in = bfi_sample_rate)

            all_file_sqi.append(SQI_final)
            file_sqi_records.append((f"{root}:{ch_name}", float(SQI_final)))

        out_dir = "./data/ppg_reports"  # The first layer of saved csv
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
        i = 0

    '''except Exception as e:
        print(f"[ERROR] in {ldf_path}\n{e}")
        continue'''

# Calculate the mean SQI across all files
# if len(all_file_sqi) > 0:
#     sqi_mean = float(np.mean(all_file_sqi))
#     print("\n==================== Results: ====================")
#     for (r, s) in file_sqi_records:
#         print(f"{r}  ->  SQI={s:.4f}")
#     print(f"\nTotal: {len(all_file_sqi)} Files, The mean of SQI  = {sqi_mean:.4f}")
# else:
#     print("\nNo SQI was successfully calculated for any file.")
