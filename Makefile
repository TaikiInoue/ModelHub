docker_run:
				docker run --runtime nvidia -it --privileged \
						--network host \
						--volume $(PWD):/app/ModelHub \
						--workdir /app \
						--name docker-xavier \
						--hostname docker-xavier \
						taikiinoue45/jetson:xavier \
						/bin/bash

install_nablasian:
				git clone git@github.com:nablas-inc/nablasian.git \
				cd /app/nablasian \
				git checkout 03eda2542ef04847bbe1a0905732602f484fc2d5 \
				pip install .

download_pth:
				cd /app/nablasian
				dvc pull detection/free_anchor/*.pth.dvc
