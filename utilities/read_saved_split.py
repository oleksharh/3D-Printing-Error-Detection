import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../clean_src")))


def compare_datasets(first, second):
    subset_1 = torch.load(first)
    print(subset_1.indices[:10], end="\n\n")    
    subset_2 = torch.load(second)
    print(subset_2.indices[:10])
    if subset_1.indices == subset_2.indices:
        print("Datasets are the same!")
        return True
    else:
        print("Datasets are different!")
        return False
    
compare_datasets("C:\\FYP\\data\\initial_layer_dataset\\test.pt", "C:\\FYP\\data_test\\test\\initial_layer_dataset\\test.pt")


#   subset_1 = torch.load(first)
# [31364, 51737, 1216, 11244, 53143, 30622, 15891, 60317, 48193, 9172]

# c:\FYP\utilities\read_saved_split.py:33: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   subset_2 = torch.load(second)
# [31364, 51737, 1216, 11244, 53143, 30622, 15891, 60317, 48193, 9172]
# Datasets are the same!