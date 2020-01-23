# -*- coding: utf-8 -*-

"""
@Author: https://github.com/nnkhoa/text2ImgDoc
Modified by @EmanuelaBoros (support chinese and greek)
"""

import os
import re
import sys
import codecs


def output_content(content, output_file, writing_type='w+'):
    with open(output_file, writing_type) as f:
        f.write(content)


def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
        print("Directory " + dir_name + " created")
    except FileExistsError:
        print("Directory " + dir_name + " existed")


def remove_tag(input_file, output_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        text = f.read().strip()
    notag_text = re.sub('<.*?>', '', text)
    output_content(notag_text, output_file, 'w+')


def txt_to_img(input_file, output_file):
    print(input_file)
    with codecs.open(input_file, 'r', 'utf-8') as f:
        text = f.read().strip()

    command = 'convert -size 1240x -fill black -pointsize 24 -fill black -verbose -font Arial-Unicode-Regular.ttf '.split(
        ' ')

    text = text.replace("'", ' ')
    text = text.replace('"', ' ')
    command.append('caption:"' + text + '"')
    command.append(output_file)

    os.popen(' '.join(command))
#    stream.read()

#TODO: subprocess too slow
#    p = subprocess.Popen(command, stdout=subprocess.PIPE)
#    output, error = p.communicate()
#    print(output)
#
#    if p.returncode != 0:
#        output_content(error, "log", 'a+')


def walk_through_dir(in_dir, out_dir, convert_func):
    for dir_path, dir_names, file_names in os.walk(in_dir):
        if len(dir_names) > 0:
            for dir_name in dir_names:
                sub_out_dir = os.path.join(out_dir, dir_path.replace('/'.join(in_dir.split('/')[:-1]), '')[1:], dir_name)
                create_dir(sub_out_dir)
        else:
            sub_out_dir = os.path.join(out_dir, dir_path.replace('/'.join(in_dir.split('/')[:-1]), '')[1:])
            convert_files(dir_path, sub_out_dir, convert_func)


def convert_files(input_dir, output_dir, convert_func):
    for doc_file in os.listdir(input_dir):
        if doc_file.startswith('.'):
            continue

        infile = os.path.join(os.path.abspath(input_dir), doc_file)

        file_extension = ''
        if convert_func.__name__ == "txt_to_img":
            file_extension = ".png"

        outfile = os.path.join(os.path.abspath(
            output_dir), doc_file + file_extension)

        convert_func(infile, outfile)


if __name__ == "__main__":
    doc_dir = sys.argv[1]
    out_dir = "output"
    notag_dir = out_dir + "/notag"
    png_dir = out_dir + "/png"

    create_dir(out_dir)
    create_dir(notag_dir)
    create_dir(png_dir)

    create_dir(notag_dir + "/" + os.path.basename(doc_dir))
    create_dir(png_dir + "/" + os.path.basename(doc_dir))

    if os.path.exists(doc_dir):
        walk_through_dir(doc_dir, notag_dir, remove_tag)
        walk_through_dir(os.path.join(
            notag_dir, os.path.basename(doc_dir)), png_dir, txt_to_img)
