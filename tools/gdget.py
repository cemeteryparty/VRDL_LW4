#!/usr/bin/python3
import requests
import argparse
import sys

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Download shared google drive file."
    )

    parser.add_argument(
        "file_id", help="Shared file id in url."
    )
    parser.add_argument(
        "-O", "--output",
        help="Download Path",
        type=str, default="./download.zip"
    )

    return parser.parse_args(args)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # print(args)
    # exit(0)
    
    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"
    file_id = args.file_id

    session = requests.Session()
    response = session.get(DOWNLOAD_URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(DOWNLOAD_URL, params=params, stream=True)

    with open(args.output, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    main()
