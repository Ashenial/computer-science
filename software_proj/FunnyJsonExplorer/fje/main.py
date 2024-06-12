from .builder import StyleBuilder
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Funny Json Explorer')
    parser.add_argument('-f', '--file', type=str, help='Json file', default='test/test.json')
    parser.add_argument('-s', '--style', type=str, help='style', choices=['tree', 'rect'], default='tree')
    parser.add_argument('-i', '--icon', type=str, help='icon', default='void')
    return parser.parse_args()

def main():
# if __name__ == '__main__':
    try:
        args = parse_args()
        builder = StyleBuilder()
        if args.icon not in builder._icons and args.icon is not None:
            icons_file = 'add_icons/' + args.icon + '.json'
            builder.load_icons(icons_file)
        builder.create_style(args.file, args.icon, args.style).plot()
    except Exception as e:
        print(f'Error: {e}')
    