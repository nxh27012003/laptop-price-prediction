import sys
import os
import pandas as pd
from fuzzywuzzy import fuzz

PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARENT_PATH = os.path.abspath(os.path.join(PARENT_PATH, '..', '..'))
DATA_PATH = os.path.abspath(os.path.join(PARENT_PATH, "data", "raw", "cpu_gpu_mark"))

CPU_FILENAME = "cpu_mark.csv"
GPU_FILENAME = "gpu_mark.csv"

cpu_df = pd.read_csv(os.path.abspath(os.path.join(DATA_PATH, CPU_FILENAME)), index_col = 0)
gpu_df = pd.read_csv(os.path.abspath(os.path.join(DATA_PATH, GPU_FILENAME)), index_col = 0)

cpu_name_list = [cpu_df['CPU Name'][idx].lower() for idx in cpu_df.index]
gpu_name_list = [gpu_df['GPU Name'][idx].lower() for idx in gpu_df.index]


def mapping(s, slist):
    s2 = s.lower()
    found=[0, -1]
    for i in range(len(slist)):
        if s2 in slist[i]:
            found[0] = 100
            found[1] = i
            break
        acc=sum([fuzz.ratio(slist[i], s2),fuzz.partial_ratio(slist[i], s2),fuzz.token_sort_ratio(slist[i], s2),fuzz.token_set_ratio(slist[i], s2)])/4

        if found[0]<acc:
            found[0] = acc
            found[1] = i
    return found

def get_cpu_name(cpu):
    acc , pos = mapping(cpu,cpu_name_list)
    return cpu_df['CPU Name'][pos], cpu_df['CPU Rank'][pos]

def get_gpu_name(gpu):
    acc, pos = mapping(gpu,gpu_name_list)
    return gpu_df['GPU Name'][pos], gpu_df['GPU Rank'][pos]

if __name__ == '__main__':
    TEST_CPU_NAME = 'Intel core i5 1155g7'

    b, cpu_mark = get_cpu_name(TEST_CPU_NAME)
    print(cpu_mark)
    print(b)
