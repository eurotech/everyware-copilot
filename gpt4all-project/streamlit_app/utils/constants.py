# constants.py
import os

#DOC_ROOT_PATH = '/opt/gpt4all-project/Documents'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
DOC_ROOT_PATH = os.path.join(grandparent_dir, 'Documents')
INDEX_ROOT_PATH = os.path.join(grandparent_dir, 'Indexes')


SUPPORTED_FILE_TYPES = ['txt', 'csv', 'docx', 'epub', 'hwp', 'ipynb', 
                        'jpeg', 'jpg', 'mbox', 'md', 'mp3', 
                        'mp4', 'pdf', 'png', 'ppt', 'pptm', 'pptx']
