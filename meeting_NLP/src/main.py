import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Intelligence Poem and Lyric Writer.')

    help_ = 'you can set this value in terminal --write value can be poem or lyric.'
    parser.add_argument('-w', '--write', default='poem', choices=['poem', 'lyric'], help=help_)

    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)
    parser.set_defaults(train=True)

    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()
    if args.write == 'poem':
        from inference import tang_poems
        if args.train:
            tang_poems.main(True)
        else:
            tang_poems.main(False)
    elif args.write == 'lyric':
        from inference import song_lyrics
        print(args.train)
        if args.train:
            song_lyrics.main(True)
        else:
            song_lyrics.main(False)
    else:
        print('[INFO] write option can only be poem or lyric right now.')




