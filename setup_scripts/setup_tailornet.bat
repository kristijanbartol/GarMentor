@echo off
setlocal enabledelayedexpansion

set argCount=0
for %%x in (%*) do (
	set /A argCount+=1
	set "argVec[!argCount!]=%%~x"
)

if NOT %argCount%==2 (
	echo "1st argument: Path to garmentor directory"
	echo "2nd argument: Path to <tailornet_data_root>"
) else (
	docker run -it --rm --gpus=0 -v %1:/garmentor -v %2:/tailornet_data -w /garmentor/tailornet_dataset_for_garmentor %username%-garmentor /bin/bash -c "
	python -m smpl_lib.convert_smpl_models
	for file in /tailornet_data/smpl/*
	do
		if [[ $file == *"smpl_"* ]]; then
			chmod a+rwx $file
		fi
	done
	"
)