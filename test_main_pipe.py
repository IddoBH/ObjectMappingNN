from sys import argv

from main_pipe import main

if __name__ == '__main__':
    main(model_path=argv[1], image_path=argv[2], output_path=argv[3])
