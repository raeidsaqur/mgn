import os, sys, platform

PROJECT_PATH=".."
CLEVR_PARSER_PATH = f'{PROJECT_PATH}/vendors/clevr-parser'
if CLEVR_PARSER_PATH not in sys.path:
    sys.path.insert(0, CLEVR_PARSER_PATH)


