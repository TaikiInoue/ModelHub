# ModelHub


Download this repository
```
git clone https://github.com/TaikiInoue/ModelHub.git
cd ModelHub
```

Run docker container
```
make docker_run
```

Install nablasian
```
make install_nablasian
```

Download pth file for backbone and object detection model
```
cd /app/ModelHub
dvc remote modify --local somic connection_string "[connection_string]"
make download_pth
```

Run FreeAnchor
```
make run_free_anchor
```
