# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
from tqdm import tqdm
import codecs
import re


def generate_images(files, level='images'):
    #    if os.path.exists(args.output_directory):
    #        shutil.rmtree(args.output_directory)
    #        os.makedirs(args.output_directory)

    for file in files:
        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                'sent', 'anchor_text', 'anchors', 'language'])

        original_filename = file

        file_name = original_filename.replace(
            '.txt', '_for_ocr_0.txt').replace('data', args.output_directory + '/text')
        file_names = [file_name]
        if os.path.exists(file_name):
            os.remove(file_name)
        
        max_lengths = []
        max_length = 0
        for idx, line in tqdm(df.iterrows(), total=len(df)):
            
            sentence = line.sent.strip()
            with open(file_name, 'a') as f:
                f.write(sentence + '\n')
                if len(sentence) > max_length:
                    max_length = len(sentence)
            if (idx+1) % 30 == 0:
                idx += 1
                file_name = original_filename.replace(
                    '.txt', '_for_ocr_' + str(idx) + '.txt').replace('data', args.output_directory + '/text')
                file_names.append(file_name)
                if os.path.exists(file_name):
                    os.remove(file_name)
                max_lengths.append(max_length)
                max_length = 0
        
        max_lengths.append(max_length)
        print(len(file_names), len(max_lengths))
#        print(max_lengths)
        for file_name, max_length in zip(file_names, max_lengths):

            #            encoded_image_path = text_to_image.encode_file(file_name, file_name.replace('.txt', '.png').replace('text', 'images'))
            if os.path.exists(file_name):
                with codecs.open(file_name, 'r') as f:
                    text = f.read().strip()

#            encoded_image_path = text_to_image.encode('hello', file_name.replace('.txt', '.png').replace('text', 'images'))
##31343
#            if max_length >= 280:
#                command = 'convert -size 3450x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            if max_length == 322:
#                command = 'convert -size 3800x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            elif max_length >= 270:
#                command = 'convert -size 2950x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            elif max_length <= 236:
#                command = 'convert -size 2500x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            elif max_length in [287, 291]:
#                command = 'convert -size 2700x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            else:
#                command = 'convert -size 2900x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
                
#            if max_length == 287:
#                command = 'convert -size 3200x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            if max_length == 318:
#                command = 'convert -size 3500x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            if max_length == 291:
#                command = 'convert -size 3200x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            if max_length == 305:
#                command = 'convert -size 3200x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            if max_length == 283:
#                command = 'convert -size 3100x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
#            if max_length == 297:
#                command = 'convert -size 3400x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
#                    ' ')
            if max_length == 330:
                command = 'convert -size 3700x -fill black -pointsize 24 -fill black -verbose -font /home/eboros/projects/text2ImgDoc/Arial-Unicode-Regular.ttf '.split(
                    ' ')#convert -units PixelsPerInch ../../data/ace_2005/images/test_for_ocr_780.png -density 300 ../../data/ace_2005/images/test_for_ocr_780.png
    
                text = text.replace("'", ' ')
                text = text.replace('"', ' ')
                
                print(file_name, len(text), max_length)
                
                command.append('caption:"' + text + '"')
                command.append(file_name.replace(
                    '.txt', '.png').replace('text', 'images'))
                
    #            import pdb;pdb.set_trace()
                os.popen(' '.join(command))

#def generate_ocr(files, directory='images_bleed_plus_character_degradation_phantom_blur_plus_plus', level='Bleed_0CharDeg_0Phantom_VERY_FREQUENT_0Blur_Complete_0'):
#
#    for file in files:
#        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
#                                                'sent', 'anchor_text', 'anchors', 'language'])
#
#        original_filename = file
#        file_name = original_filename.replace('.txt', '_for_ocr_0' + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
#        file_names = [file_name]
#        
##        import pdb;pdb.set_trace()
#
#        for idx, line in tqdm(df.iterrows(), total=len(df)):
#            if (idx+1) % 30 == 0:
#                idx += 1
#                file_name = original_filename.replace(
#                    '.txt', '_for_ocr_' + str(idx) + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
#                file_names.append(file_name)
#
##        import pdb;pdb.set_trace()
#        for file_name in file_names:
#            print('tesseract ' + file_name + ' ' + file_name.replace(directory, 'ocr_' + directory).replace('.png', '') + ' -l eng')
##            os.popen('tesseract ' + file_name + ' ' + file_name.replace(directory, 'ocr_' + directory).replace('.png', '') + ' -l eng')
def generate_ocr(files, directory='images_bleed_plus_character_degradation_phantom_blur_plus_plus', level='Bleed_0CharDeg_0Phantom_VERY_FREQUENT_0Blur_Complete_0'):

    for file in files:
        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                'sent', 'anchor_text', 'anchors', 'language'])

        original_filename = file
        file_name = original_filename.replace('.txt', '_for_ocr_0' + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
        file_names = [file_name]
        
#        import pdb;pdb.set_trace()

        for idx, line in tqdm(df.iterrows(), total=len(df)):
            if (idx+1) % 30 == 0:
                idx += 1
                file_name = original_filename.replace(
                    '.txt', '_for_ocr_' + str(idx) + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
                file_names.append(file_name)

#        import pdb;pdb.set_trace()
        for file_name in file_names:
            print('tesseract ' + file_name + ' ' + file_name.replace(directory, 'ocr_' + directory).replace('.png', '') + ' -l eng')
#            os.popen('tesseract ' + file_name + ' ' + file_name.replace(directory, 'ocr_' + directory).replace('.png', '') + ' -l eng')


def align_ocr(files, directory='images_character_degradation', level='CharDeg_0'):
    for file in files:
        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                'sent', 'anchor_text', 'anchors', 'language'])

        original_filename = file
        file_name = original_filename.replace('.txt', '_for_ocr_0' + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
        file_names = [file_name]

        for idx, line in tqdm(df.iterrows(), total=len(df)):
            if (idx+1) % 30 == 0:
                idx += 1
                file_name = original_filename.replace(
                    '.txt', '_for_ocr_' + str(idx) + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
                file_names.append(file_name)

        for file_name in file_names:
            command = 'java RecursiveAlignmentTool ' + file_name + ' ' + file_name.replace('.txt', '_ocr1.txt').replace('text', 'ocr1') + ' ' + \
                file_name.replace('.txt', '_alignment_ocr1.txt').replace(
                    'text', 'align_ocr1') + ' -opt config.txt'
            print(command)
            os.popen(command)


def make_train_test(files, directory='ocr_images_bleed_plus_character_degradation_phantom_blur_plus_plus', level='Bleed_0CharDeg_0Phantom_VERY_FREQUENT_0Blur_Complete_0'):
    for file in files:
        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                'sent', 'anchor_text', 'anchors', 'language'])

        original_filename = file
        file_name = original_filename.replace('.txt', '_for_ocr_0' + level + '.txt').replace('ace_2005', 'ace_2005/' + directory)
        file_names = [file_name]
        
        original_text_lines = []
        for idx, line in tqdm(df.iterrows(), total=len(df)):
            original_text_lines.append(line.sent.strip())
            if (idx+1) % 30 == 0:
                idx += 1
                file_name = original_filename.replace(
                    '.txt', '_for_ocr_' + str(idx) + level + '.txt').replace('ace_2005', 'ace_2005/' + directory)
                file_names.append(file_name)
                
#        import pdb;pdb.set_trace()

        ocr_text_lines = []
        for file_name in file_names:
#            file_name = file_name.replace(
#                '.txt', '_ocr1.txt').replace('text', 'ocr1')
            with open(file_name, 'r') as f:
                text_lines = f.readlines()

#            import pdb;pdb.set_trace()

            for text_line in text_lines:
                text_line = text_line.replace('\n', '').strip()
                if len(text_line) > 0:
                    text_line = re.sub(r"\s+", ' ', text_line)
                    ocr_text_lines.append(text_line)

        df = pd.read_csv(original_filename, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                             'sent', 'anchor_text', 'anchors', 'language'])
    
        
        for idx, line in tqdm(df.iterrows(), total=len(df)):
            df.loc[idx, 'sent'] = ocr_text_lines[idx].replace('‘', ' ').replace("'", ' ')
        
        from ast import literal_eval
        df.to_csv(original_filename.replace('.txt', '_ocr_' + level +'.txt'), sep='\t', index=False)
#        
        df = pd.read_csv(original_filename.replace('.txt', '_ocr_' + level +'.txt'), sep='\t', converters={"anchors": literal_eval})#, converters={"anchors": literal_eval}
#        import pdb;pdb.set_trace()#, names=['doc_id',  'type', 'subtype', 'blabla', 'sent', 'anchor_text', 'anchors', 'language']

        for text_line, original_sentence in zip(ocr_text_lines, original_text_lines):
            print('-'*30)
            print(text_line)
            print(original_sentence)

def remove_double_enter(files, directory='ocr_images_bleed_plus_character_degradation_phantom_blur_plus_plus', level='Bleed_0CharDeg_0Phantom_VERY_FREQUENT_0Blur_Complete_0'):

    for file in files:
        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                'sent', 'anchor_text', 'anchors', 'language'])

        original_filename = file
        file_name = original_filename.replace('.txt', '_for_ocr_0' + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
        file_names = [file_name]
        
#        import pdb;pdb.set_trace()

        for idx, line in tqdm(df.iterrows(), total=len(df)):
            if (idx+1) % 30 == 0:
                idx += 1
                file_name = original_filename.replace(
                    '.txt', '_for_ocr_' + str(idx) + level + '.png').replace('ace_2005', 'ace_2005/' + directory)
                file_names.append(file_name)

#        import pdb;pdb.set_trace()
        for file_name in file_names:
            file_name = file_name.replace(directory, directory).replace('.png', '.txt')
            print(file_name)
            with codecs.open(file_name, 'r') as f:
                text = f.read().strip()
            
            text = text.replace('\n\n','\n')
            with open(file_name, 'w') as f:
                f.write(text)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trigger pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('--test',
                        help='Test file')
    parser.add_argument('--valid',
                        help='Validation file')
    parser.add_argument('--augment', default=False,
                        help='Augment or not')
    parser.add_argument('--output_directory', default='data',
                        help='Augment or not')

    args = parser.parse_args()

    files = [args.test]
    assert os.path.exists(args.test)
#    assert os.path.exists(args.valid)

#    generate_images(files)
#    align_ocr(files)
    generate_ocr(files)
    remove_double_enter(files)
    make_train_test(files)
