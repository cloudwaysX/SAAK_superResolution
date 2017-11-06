# echo '============ 128--> 64  =========' >> BSD_scale_4.txt

echo '============ cluster 5  =========' >> BSD_scale_4.txt
python lsMapping_patch.py hierachy_kmean --cluster_num 5 --scale 4 --LR_sample_folder _BSD_0_0_1e-5 --HR_sample_folder _BSD_0_0_1e-5 --HR_weight_folder BSD_weight_0_0_1e-5 --LR_stop_layer _L3_1e-05 --HR_stop_layer _L3_1e-05  >> BSD_scale_4.txt

echo '============ cluster 10  =========' >> BSD_scale_4.txt
python lsMapping_patch.py hierachy_kmean --cluster_num 10 --scale 4 --LR_sample_folder _BSD_0_0_1e-5 --HR_sample_folder _BSD_0_0_1e-5 --HR_weight_folder BSD_weight_0_0_1e-5 --LR_stop_layer _L3_1e-05 --HR_stop_layer _L3_1e-05  >> BSD_scale_4.txt

# echo '============ cluster 20  =========' >> BSD_scale_4.txt
# python lsMapping_patch.py hierachy_kmean --cluster_num 20 --scale 4 --LR_sample_folder _BSD_0_0_1e-5 --HR_sample_folder _BSD_0_0_1e-5 --HR_weight_folder BSD_weight_0_0_1e-5 --LR_stop_layer _L3_1e-05 --HR_stop_layer _L3_1e-05  >> BSD_scale_4.txt

# echo '============ cluster 30  =========' >> BSD_scale_4.txt
# python lsMapping_patch.py hierachy_kmean --cluster_num 30 --scale 4 --LR_sample_folder _BSD_0_0_1e-5 --HR_sample_folder _BSD_0_0_1e-5 --HR_weight_folder BSD_weight_0_0_1e-5 --LR_stop_layer _L3_1e-05 --HR_stop_layer _L3_1e-05  >> BSD_scale_4.txt

# echo '============ cluster 40  =========' >> BSD_scale_4.txt
# python lsMapping_patch.py hierachy_kmean --cluster_num 40 --scale 4 --LR_sample_folder _BSD_0_0_1e-5 --HR_sample_folder _BSD_0_0_1e-5 --HR_weight_folder BSD_weight_0_0_1e-5 --LR_stop_layer _L3_1e-05 --HR_stop_layer _L3_1e-05  >> BSD_scale_4.txt

# echo '============ cluster 50  =========' >> BSD_scale_4.txt
# python lsMapping_patch.py hierachy_kmean --cluster_num 50 --scale 4 --LR_sample_folder _BSD_0_0_1e-5 --HR_sample_folder _BSD_0_0_1e-5 --HR_weight_folder BSD_weight_0_0_1e-5 --LR_stop_layer _L3_1e-05 --HR_stop_layer _L3_1e-05  >> BSD_scale_4.txt
