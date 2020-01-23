Project forked from https://github.com/nnkhoa/text2ImgDoc. Modified to support Unicode.

# Text To Image Documents script

Requirements: ImageMagick 

### How to run: 

```python text2img.py DATA_FOLDER```

```DATA_FOLDER``` is the directory containing all of the text files as well as sub-directories (two or more levels of sub-folder may cause crashes. It's best to not have any sub-directory at all).

This will create two new folders: One containing the clean text file without any Html tags, the other having all of the text files converted into images.

Note: Removing Html tags currently only works with articles from DAniEL dataset. Just comment out line #82 if you already have a clean dataset.
