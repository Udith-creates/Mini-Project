from pathlib import Path
import sys

try:
    from pytube import YouTube
except Exception as e:
    print("Missing dependency: pytube. Install it in your venv with:\n    python -m pip install pytube")
    raise


def download(url: str, out_dir: str = "data/test"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    yt = YouTube(url)
    # get the highest-resolution progressive stream (audio+video)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if stream is None:
        raise RuntimeError("No suitable mp4 progressive stream found for this video")

    target = out / f"{yt.video_id}.mp4"
    print(f"Downloading: {yt.title}\nSaving to: {target}")
    stream.download(output_path=str(out), filename=f"{yt.video_id}.mp4")
    print("Done")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python download_youtube.py <youtube_url>")
        sys.exit(1)
    download(sys.argv[1])
