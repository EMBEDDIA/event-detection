NOTES:
- The core algorithm of Daniel still being coded in Python2.7, you will have to change the commands below if Python3.x is your default version
- Using the pypy compiler usually makes Daniel 10 times faster (see https://pypy.org/download.html and choose the appropriate Python2.7 version)
- On a standard laptop, in standard conditions for temperature and pressure, Daniel processes around 100 documents/second with the pypy2 compiler, around 10 with the Python2.7 compiler


[daniel.py]  For testing simple files
  Can be tested with the following command (-v to print results on stdout):
    python daniel.py -l LANGUAGE -d FILE -v
  Example :
    python daniel.py -l id -d some_document_in_indonesian.html -v

[process_corpus.py]  For processing a corpus :
  The command :
    python process_corpus.py -c JSON_FILE
      NB: Needs a JSON file  (see below for the format)
  Example :
     python2.7 process_corpus.py -c docs/Indonesian_GL.json

[evaluate.py] For evaluating results
  Compares the content of a groundtruth JSON file and an output from daniel
    python evaluate.py GROUNDTRUTH DANIEL_OUTPUT

[The JSON format]
  A dictionnary where each key is the ID of a document
  The value is a dictionnary with informations on the document:  
    - mandatory information :
      - file path
    - useful informations (by decreasing importance) :
      - source
      - language
      - url
      - comment
    -information for evaluation :
      - annotations
    See docs/Indonesian_GL.json for an example

