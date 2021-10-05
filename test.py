import utils.helpful_functions as hf
import utils.ta_lib_indicators as ti

print(hf.ma_types_dict)

for f in hf.ma_types_list:
    print(f'{ti.overlap_studies[f]}')

print(hf.choose_ma_type())
