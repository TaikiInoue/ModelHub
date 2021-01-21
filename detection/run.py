from ftplib import FTP
from pathlib import Path
from time import sleep
from typing import List

from free_anchor import FreeAnchor


class Inspector:
    def __init__(self) -> None:

        self.model = FreeAnchor("./free_anchor/config.yaml")
        self.ftp = FTP(host="10.3.1.3", user="somic", passwd="somic")
        self.local_base = Path(".")
        self.remote_base = Path(".")
        self.img_suffix = "jpg"
        self.done_filename: List[str] = []

        print("DONE: Initialize")

    def start_deamon(self) -> None:

        while True:
            for remote_filename in self.ftp_ls():
                sleep(1)
                if remote_filename not in self.done_filename:
                    self.done_filename.append(remote_filename)
                    stem = Path(remote_filename).stem
                    self.inference(stem)
                    self.clean_up()
                else:
                    continue

    def start_debug(self) -> None:

        self.inference("free_anchor/tobu")
        self.inference("free_anchor/ziku")

    def inference(self, stem: str) -> None:

        self.ftp_get(stem)
        img = self.model.pre_processing(f"{stem}.{self.img_suffix}")
        mb_bboxes, mb_scores, mb_labels = self.model.inference(img)
        self.write_result(stem, mb_bboxes)
        self.ftp_put(stem)
        self.clean_up()

    def write_result(self, stem: str, mb_bboxes) -> None:

        result_stem = stem.split("_")[0]
        if len(mb_bboxes) == 0:
            lines = ["OK\n"]
        else:
            lines = ["NG\n"]
            for bbox in mb_bboxes.detach().cpu().numpy():
                x0, y0, x1, y1 = bbox
                lines += [f"{x0}, {y0}, {x1}, {y1}\n"]
        with open(f"{result_stem}.txt", "w") as f:
            f.writelines(lines)

    def clean_up(self) -> None:

        for p in self.local_base.glob("*.txt"):
            p.unlink()

        for p in self.local_base.glob("*.jpg"):
            p.unlink()

    def ftp_ls(self) -> List[str]:

        return self.ftp.nlst(f"*.{self.img_suffix}")

    def ftp_get(self, stem: str) -> None:

        filename = f"{stem}.{self.img_suffix}"
        local_file_path = str(self.local_base / filename)
        remote_file_path = str(self.remote_base / filename)

        with open(local_file_path, "wb") as f:
            self.ftp.retrbinary(f"RETR {remote_file_path}", f.write)

    def ftp_put(self, stem: str) -> None:

        result_stem = stem.split("_")[0]
        filename = f"{result_stem}.txt"
        local_file_path = str(self.local_base / filename)
        remote_file_path = str(self.remote_base / f"Judge/{filename}")
        with open(local_file_path, "rb") as f:
            self.ftp.storbinary(f"STOR {remote_file_path}", f)


def main() -> None:

    inspector = Inspector()
    inspector.start_debug()


if __name__ == "__main__":

    main()
