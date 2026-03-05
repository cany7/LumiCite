from pathlib import Path
import pandas as pd
import requests

COMMON_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp1252",   # Windows-1252, common on Windows
    "latin-1",  # ISO-8859-1, permissive fallback
]

def detect_encoding(file_path: Path, test_bytes: int = 4096) -> str:
    """
    Try a small read with a set of common encodings to find one that works.
    Falls back to latin-1 which never fails but may slightly misinterpret rare characters.
    """
    data = file_path.read_bytes()[:test_bytes]
    for enc in COMMON_ENCODINGS:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    # As a last resort, use latin-1 which maps bytes 1:1
    return "latin-1"

def load_metadata_df():
    """
        Loads the metadata csv into a pandas dataframe for future processing. Dynamically
        Retrieves the metadata.csv file path from the current file's location.
    """
    # start from the current file
    path = Path(__file__).resolve()

    # walk up until folder name is the project root
    while not ((path / "src").exists() and (path / "data").exists()):
        if path.parent == path:  # safety check in case we reach the filesystem root
            raise RuntimeError("Project root directory not found")
        path = path.parent

    # get the metedata and pdf directory paths from the file root
    meta_path = path / "data" / "metadata" / "metadata.csv"
    pdf_dir = path / "data" / "raw_pdfs"

    # check if both paths exist:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # create metadata dataframe
    df = pd.read_csv(meta_path, encoding="utf-8")
    print("Successfully loaded metadata csv")
    print(df.head())
    return df

def get_PDF_paths():
    """
        Returns a dictionary with key-value pairs of PDF file IDs and their paths.
    """
    # start from the current file
    path = Path(__file__).resolve()

    # walk up until folder name is the project root
    while not ((path / "src").exists() and (path / "data").exists()):
        if path.parent == path:  # safety check in case we reach the filesystem root
            raise RuntimeError("Project root directory not found")
        path = path.parent

    # get the PDF folder path from the file root
    pdf_dir = path / "data" / "pdf"
    if not pdf_dir.exists():
        print(f"Folder {pdf_dir} does not exist")
    elif pdf_dir.exists():
        print(f"Folder {pdf_dir} successfully found")

    # get all the PDF files in the folder
    files = list(pdf_dir.iterdir())
    print(f"Number of PDFs in folder: {len(files)}")
    print(f"PDF files: {files}")

    # get all path stems (IDS) from the PDF files
    stems = [p.stem for p in files]
    # get all file paths from the PDF files
    paths = [str(p) for p in files]

    pdf_dict = {}
    for i in range(len(stems)):
        pdf_dict[stems[i]] = paths[i]

    return pdf_dict


def download_pdfs(df):
    # walk up until folder name is the project root
    path = Path(__file__).resolve()
    while not ((path / "src").exists() and (path / "data").exists()):
        if path.parent == path:  # safety check in case we reach the filesystem root
            raise RuntimeError("Project root directory not found")
        path = path.parent

    # get the download folder from the file root
    pdf_dir = path / "data" / "pdf"

    # Create PDF folder if it doesn't exist
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Folder {path} does not exist. Creating folder...")

    # Check if the PDF folder is empty. Download PDFS if not
    files = list(pdf_dir.iterdir())
    if not files:
        print(f"Folder {pdf_dir} is empty. Downloading PDFs...")
        # Iterate over rows in the metadata dataframe

        for _, row in df.iterrows():
            url = row["url"]
            try:
                # attempt to retrieve the PDF from the URL
                response = requests.get(url)
                response.raise_for_status()

                # download the PDF as the ID from the metadata dataframe
                pdf_path = pdf_dir / f"{row['id']}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {row['id']}.pdf")

            except Exception as e:
                # handle any errors during the download and skip files that fail
                print(f"Failed to download {row['id']}: {e}")
                continue
        print("Finished downloading PDFs.")
    else:
        print(f"Folder {pdf_dir} is not empty. Skipping download...")

    # get the IDs from all the downloaded files & files meant to be downloaded
    downloaded = [f.stem for f in pdf_dir.iterdir() if f.is_file()]
    to_download = df["id"].tolist()

    # check if all files were downloaded. Print files that weren't downloaded
    if set(downloaded) == set(to_download):
        print("All files downloaded.")
    else:
        print("Some files were not downloaded.")
        print(set(to_download)-set(downloaded))

    # Print number of files downloaded and to download
    print(f"Number of PDFs downloaded: {len(files)}")
    print(f"Number of PDFs to download: {len(to_download)}")

    # Returns a list of file paths in the PDF folder.
    files = list(pdf_dir.iterdir())
    return files

def load_data():
    df = load_metadata_df()
    download_pdfs(df)

