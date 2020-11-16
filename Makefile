docker_run:
				docker run --runtime nvidia -it --privileged \
						--network host \
						--volume $(PWD):/app/ModelHub \
						--workdir /app/ModelHub \
						--name docker-xavier \
						--hostname docker-xavier \
						taikiinoue45/jetson:xavier \
						/bin/bash

install_nablasian:
				git clone 

download_pth:
				dvc pull detection/free_anchor/*.pth.dvc
