from free_anchor import FreeAnchor
from ftplib import FTP
from pathlib import Path


class Inspector:

    def __init__(self):

        self.model = FreeAnchor("./free_anchor/config.yaml")
        self.ftp = FTP(host="192.168.2.174", user="ftp_server", passwd="inoue")

        self.local_base = Path(".")
        self.remote_base = Path(".")
        self.img_suffix = "jpg"
        self.done_filename = []

    def start_deamon(self):

        while True:
            for remote_filename in self.ftp_ls():
                if remote_filename not in self.done_filename:
                    stem = Path(remote_filename).stem
                    self.inference(stem)
                    # self.clean_up()
                else:
                    continue

            break

    def inference(self, stem):

        self.ftp_get(stem)
        img = self.model.pre_processing(f"{stem}.{self.img_suffix}")
        mb_pre_bboxes, mb_cls_logits = self.model.inference(img)
        self.write_result(stem, mb_pre_bboxes)
        self.ftp_put(stem)
        self.clean_up()

    def write_result(self, stem, mb_pre_bboxes):

        with open(f"{stem}.txt", "w") as f:
            if len(mb_pre_bboxes) == 0:
                f.write("OK\n")
            else:
                f.write("NG\n")

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

        filename = f"{stem}.txt"
        local_file_path = str(self.local_base / filename)
        remote_file_path = str(self.remote_base / f"uploaded/{filename}")

        with open(local_file_path, "rb") as f:
            self.ftp.storbinary(f"STOR {remote_file_path}", f)


def main():

    inspector = Inspector()
    inspector.start_deamon()

if __name__ == "__main__":

    main()
