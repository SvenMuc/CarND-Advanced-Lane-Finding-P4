from moviepy.editor import ImageSequenceClip
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    parser.add_argument(
        '--overlay',
        type=str,
        dest='video_overlay',
        help='Adds a standard overlay to a video file.')


    args = parser.parse_args()

    if args.video_overlay:
        # adds a standard overlay to the video
        video = VideoFileClip(args.video_overlay)
        txt_clip = (TextClip('Udacity - Advanced Lane Finding, Sven Bone, 06.2017', font='Arial', fontsize=16, color='white')
                    .set_position(('left', 'bottom'))
                    .set_duration(50))

        result = CompositeVideoClip([video, txt_clip])
        result.write_videofile("out.mp4", fps=video.fps)

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
