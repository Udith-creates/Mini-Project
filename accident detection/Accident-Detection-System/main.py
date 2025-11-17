import argparse
from camera import startapplication


def main():
	parser = argparse.ArgumentParser(description="Run accident detection on a camera or video file")
	parser.add_argument("--video", "-v", help="Path to video file or camera index (default: 0)", default=0)
	args = parser.parse_args()

	startapplication(args.video)


if __name__ == '__main__':
	main()