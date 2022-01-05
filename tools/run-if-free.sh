#!/bin/bash

while true; do
	occ0=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
	occ1=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | wc -l)
	printf "${occ0} ${occ1}\r"
	if [ "${occ0}" -eq "0" ]; then
		echo "GPU 0 is free now      "
		export CUDA_VISIBLE_DEVICES="0"
		break
	elif [ "${occ1}" -eq "0" ]; then
		echo "GPU 1 is free now      "
		export CUDA_VISIBLE_DEVICES="1"
		break
	fi
	printf "GPU 0 has ${occ0} proc, GPU 1 has ${occ1} proc."
	sleep 5
done

make clean
python3 train.py --dataset-path training_hr_images --crop_size 160 --upscale_factor 4 \
	--epochs 200 --batch-size 32 --save-path models
exit 0
for ((i=0; i<30; i++)) do
	python3 CLEEGN_Eva_p.py ${i}
done

