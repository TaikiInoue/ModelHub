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
				cd /app && \
				git clone https://github.com/nablas-inc/nablasian.git && \
				cd /app/nablasian && \
				git checkout git checkout 0ef6a15057574068e87b9a4f637b42b3bda2cf92 && \
				python3 -m pip install .

download_pth:
				cd /app/nablasian && \
				read -sp 'connection_string: ' CONNECTION_STRING && \
				dvc remote modify --local somic connection_string "$CONNECTION_STRING" && \
				dvc pull detection/free_anchor/*.pth.dvc && \
				cd /app
