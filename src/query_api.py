import json
import logging
import os
import re
from pathlib import Path
from urllib.request import urlopen

import requests
from tqdm import tqdm

from src.common.utils import Paths, Settings


class CDRCQuery:
    def __init__(
        self,
        api_url: str,
        login_url: str,
        data_dir: Path = Paths.DATA_DIR,
        profiles_dir: Path = Paths.CDRC_PROFILES_DIR,
        login_details: dict[str, str | None] | None = None,
    ):
        if login_details is None:
            login_details = {
                "name": os.getenv("name"),
                "pass": os.getenv("pass"),
                "form_build_id": os.getenv("form_build_id"),
            }
        self.api_url = api_url
        self.login_url = login_url
        self.data_dir = data_dir
        self.profiles_dir = profiles_dir
        self.login_details = login_details

        self.profiles_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        self.catalogue_metadata = json.loads(urlopen(self.api_url).read())["result"][0]
        self.process_metadata()

        if self.check_if_files_changed():
            print("Files have changed.")
            self.files_changed = True
            self.download_files()
        else:
            print("No files have changed.")
            self.files_changed = False

    def process_metadata(self):
        self.get_metadata()
        self.file_ids = {file["id"] for file in self.files_metadata}
        self.catalogue_ids = {catalogue["id"] for catalogue in self.catalogue_metadata}
        self.write_metadata()

    def check_if_files_changed(self) -> bool:
        current_files = self.data_dir / "file_list.json"
        file_names = {
            file["name"]: file["last_modified"] for file in self.files_metadata
        }

        if not current_files.exists():
            with open(current_files, "w") as f:
                json.dump(file_names, f)
                return True
        else:
            with open(current_files) as f:
                old_response = json.load(f)
            with open(current_files, "w") as f:
                json.dump(file_names, f)
            return old_response != file_names

    def get_metadata(self):
        self.files_metadata = []
        for item in self.catalogue_metadata:
            if "resources" not in item:
                continue
            for file in item["resources"]:
                if any(x in file["name"].lower() for x in ["profile", "flyer"]):
                    file["filename"] = file["url"].split("/")[-1]
                    file["parent_id"] = item["id"]
                    self.files_metadata.append(file)

            if "notes" not in item:
                continue
            out_file = self.profiles_dir / f"notes-{item['id']}.txt"
            if out_file.exists():
                logging.info(f"Skipping {out_file} as it already exists")
                continue
            with open(out_file, "w") as f:
                f.write(
                    f"Dataset Title: {item['title']} "
                    "\n\nDescription: \n\n "
                    f"{re.sub('<[^<]+?>','', item['notes'])}"
                )

    def download_files(self):
        s = requests.Session()
        s.post(
            self.login_url,
            data={
                **self.login_details,
                "form_id": "user_login",
                "op": "Log in",
            },
        )

        for meta in tqdm(self.files_metadata, desc="Downloading files"):
            filename = (
                f"profile-{meta['id']}.{meta['format']}"
                if "profile" in meta["name"].lower()
                else f"flyer-{meta['id']}.{meta['format']}"
            )
            if (meta["url"] == "") or (self.profiles_dir / filename).exists():
                logging.info(f"Skipping {filename} as it already exists")
                continue
            file = s.get(meta["url"])
            with open(self.profiles_dir / filename, "wb") as f:
                f.write(file.content)

    def write_metadata(self) -> None:
        with open(self.data_dir / "cdrc" / "catalogue-metadata.json", "w") as f:
            json.dump(self.catalogue_metadata, f)
        with open(self.data_dir / "cdrc" / "files-metadata.json", "w") as f:
            json.dump(self.files_metadata, f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR, filename="logs/query_api.log", filemode="w"
    )

    query = CDRCQuery(**Settings().cdrc.model_dump())
    query.run()
