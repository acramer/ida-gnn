conda:
	echo [MAKE - initialize]
	conda env create -f ida.yml

add:
	echo [MAKE - add]
	conda env export --from-history | head -n -1 > ida.yml 


update:
	echo [MAKE - update]
	conda env update -f ida.yml

clean:
	echo [MAKE - clean]
	echo TODO: enable activation and deactivate of conda in this target
	conda env remove -n ida
	rm -rf code/__pycache__

test:
	echo [MAKE - test]
	cd code && python main.py --mode verify && echo 'test passed'



gpu: test setup-gpu
	echo [MAKE - gpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./DL-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

gpu-next: check-gpu
	echo [MAKE - gpu-soft-kill]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./DL-project/scripts/cloud_next.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

over-gpu: setup-gpu
	echo [MAKE - over-gpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./DL-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

under-gpu:
	echo [MAKE - under-gpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./DL-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

setup-gpu:
	echo [MAKE - gpu]
	git diff-index --quiet HEAD code || (git add code && git commit -m 'gpu commit' && git push)
	make gpu-soft-kill || echo "CPU already off"
	gcloud compute instances start alan

check-gpu:
	echo [MAKE - check-gpu]
	gcloud compute instances list | python scripts/parse_instances.py -m alan

gpu-kill: check-gpu
	echo [MAKE - gpu-kill]
	gcloud compute instances stop alan

gpu-soft-kill: check-gpu
	echo [MAKE - gpu-soft-kill]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh alan --command 'nohup ./DL-project/scripts/cloud_kill.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done




cpu: test setup-cpu
	echo [MAKE - cpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh hopper --command 'nohup ./DL-project/scripts/cloud_script.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

cpu-next: check-cpu
	echo [MAKE - gpu-soft-kill]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh hopper --command 'nohup ./DL-project/scripts/cloud_next.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

predict-cpu: setup-cpu
	echo [MAKE - predict-cpu]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh hopper --command 'nohup ./DL-project/scripts/cloud_predict.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done

setup-cpu:
	echo [MAKE - gpu]
	git diff-index --quiet HEAD code || (git add code && git commit -m 'gpu commit' && git push)
	make cpu-kill || echo "CPU already off"
	gcloud compute instances start hopper

check-cpu:
	echo [MAKE - check-cpu]
	gcloud compute instances list | python scripts/parse_instances.py -m hopper

cpu-kill: check-cpu
	echo [MAKE - cpu-kill]
	gcloud compute instances stop hopper

cpu-soft-kill: check-cpu
	echo [MAKE - gpu-soft-kill]
	for n in 1 2 3 4 5 ; do \
	gcloud compute ssh hopper --command 'nohup ./DL-project/scripts/cloud_kill.sh &> log.txt < /dev/null &' && echo 'SSH successful' && break; \
	sleep 5; \
	done
