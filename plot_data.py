import matplotlib.pyplot as plt
import pandas as pd
from util_module import SCADA_utils
from util_module.data_to_plot import visualize_events
import matplotlib

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'


# download data
dataset_name = "haenkaze"
data_path = "/mnt/work-qnap/miyauchi"
target_col = " ﾋﾟｯﾁ ﾓｰﾀｰ電流 3 (A)"

# preprocessing
target_file = f'{data_path}/data/{dataset_name}/fixed_data/2023_fixed.parquet'
print(f"Loading data from {target_file}...")
data = pd.read_parquet(target_file)
stampcol = "DateTime"
range = ('2023-06-01','2023-07-31')
data = SCADA_utils.extract_specific_terms(data,range[0],range[1],stampcol)
X = data[target_col]
timestamp = data[stampcol]
print("Finished loading data.")

df = pd.DataFrame({"Datetime": timestamp, "Data": X})
event_files = ["events.csv"]
#event_files.append(f"{data_path}/data/haenkaze/event_range.csv")
img_path = "data.png"
plt.figure(figsize=(32, 10))
for file in event_files:
    visualize_events(file)
plt.scatter(df["Datetime"], df["Data"], alpha=0.8, marker='o', color='blue')
plt.xlabel("Date",fontsize=40)
plt.ylabel(target_col,fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.grid(True)
plt.savefig(img_path)
print(f"Saved plot to {img_path}")