from free_anchor import FreeAnchor
from ftplib import FTP
from pathlib import Path
from time import time
import sys
from time import sleep


class Inspector:

    def __init__(self):

        self.model = FreeAnchor("./free_anchor/config.yaml")
        self.ftp = FTP(host="10.3.1.3", user="somic", passwd="somic")

        self.local_base = Path(".")
        self.remote_base = Path(".")
        self.img_suffix = "jpg"
        self.done_filename = []
        self.time_dict = {"ftp_get": 0, "pre_processing": 0, "inference": 0, "write_result": 0, "ftp_put": 0, "clean_up": 0}
        print("DONE: Initialize")

    def start_deamon(self):

        while True:
            for remote_filename in self.ftp_ls():
                sleep(1)
                if remote_filename not in self.done_filename:
                    self.done_filename.append(remote_filename)
                    stem = Path(remote_filename).stem
                    self.inference(stem)
                    # self.clean_up()
                else:
                    continue
                
            for k, v in self.time_dict.items():
                print(f"{k}: {v}")
            
            break


    def inference(self, stem):
        
        t0 = time()
        self.ftp_get(stem)
        t1 = time()
        img = self.model.pre_processing(f"{stem}.{self.img_suffix}")
        t2 = time()
        mb_pre_bboxes, mb_cls_logits = self.model.inference(img)
        t3 = time()
        self.write_result(stem, mb_pre_bboxes)
        t4 = time()
        self.ftp_put(stem)
        t5 = time()
        self.clean_up()
        t6 = time()
        
        self.time_dict["ftp_get"] += t1 - t0
        self.time_dict["pre_processing"] += t2 - t1
        self.time_dict["inference"] += t3 - t2
        self.time_dict["write_result"] += t4 - t3
        self.time_dict["ftp_put"] += t5 - t4
        self.time_dict["clean_up"] += t6 - t5

    def write_result(self, stem, mb_pre_bboxes):

        result_stem = stem.split("_")[0]
        if len(mb_pre_bboxes) ==0:
            lines = ["OK\n"]
        else:
            lines = ["NG\n"]
            for bbox in mb_pre_bboxes.detach().cpu().numpy():
                x0, y0, x1, y1 = bbox
                lines += [f"{x0}, {y0}, {x1}, {y1}\n"]
        with open(f"{result_stem}.txt", "w") as f:
            f.writelines(lines)
        """ 
        lines = []
        if len(mb_pre_bboxes):
            for bbox in mb_pre_bboxes.detach().cpu().numpy():
                print(bbox)
                x0, y0, x1, y1 = bbox
                lines += [f"{x0}, {y0}, {x1}, {y1}\n"]
       
        with open(f"/app/nas/result/{stem}.txt", "w") as f:
            f.writelines(lines)
        """

    def clean_up(self):

        for p in self.local_base.glob("*.txt"):
            p.unlink()

        for p in self.local_base.glob("*.jpg"):
            p.unlink()

    def ftp_ls(self):

        return self.ftp.nlst(f"*.{self.img_suffix}")

    def ftp_get(self, stem):
        
        filename = f"{stem}.{self.img_suffix}"
        local_file_path = str(self.local_base / filename)
        remote_file_path = str(self.remote_base / filename)

        with open(local_file_path, "wb") as f:
            self.ftp.retrbinary(f"RETR {remote_file_path}", f.write)

    def ftp_put(self, stem):
        
        result_stem = stem.split("_")[0]
        filename = f"{result_stem}.txt"
        local_file_path = str(self.local_base / filename)
        remote_file_path = str(self.remote_base / f"Judge/{filename}")
        with open(local_file_path, "rb") as f:
            self.ftp.storbinary(f"STOR {remote_file_path}", f)

def main():

    inspector = Inspector()
    inspector.start_deamon()

if __name__ == "__main__":

    main()
