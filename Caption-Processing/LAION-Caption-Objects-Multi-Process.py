# How to run?
# 1) Install Dependencies
# !pip install stanza
# !pip install ftfy regex tqdm
# !pip install datasets==2.9
# 2) Run in Terminal
# !python LAION-Caption-Objects-Multi-Process.py -n <NUMBER OF NODES> -g <NUMBER OF GPUS PER NODE>
# 3) Want to know more options available?
# Check the main function below for the possible arguments

# General
import os
import gc
import json
import time
import argparse
from tqdm import tqdm

# Data-Handling
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

# Distributed Training/Inference
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Model Handling
import torch
import torch.nn as nn

# Caption-Processing
from nltk.corpus import stopwords
import nltk

# POS-Tagging
import stanza

# For processing data in batches
def group_batch(batch):
  return {k: [v] for k, v in batch.items()}

# extract parts of speech
def extract_pos(doc):
  parsed_text = list()
  for sent in doc.sentences:
    parsed_sent = list()
    for wrd in sent.words:
      #extract text and pos
      parsed_sent.append((wrd.text, wrd.xpos))
    parsed_text.append(parsed_sent)
  return parsed_text

# extract lemma
def extract_lemma(doc):
  parsed_text = list()
  for sent in doc.sentences:
    parsed_sent = list()
    for wrd in sent.words:
      # extract text and lemma
      parsed_sent.append((wrd.text, wrd.lemma))
    parsed_text.append(parsed_sent)
  return parsed_text

# cleans a list of prompt
def clean_prompt(sentences, nlp):

  # treebank-specific POS (XPOS) tags to keep, other POS tagged tokens will not be retained
  keep_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']

  # Stopwords
  stpwords = set(stopwords.words('english'))

  # convert the sentences to lower case
  sentences_lc = [sentence.lower() for sentence in sentences]

  # stanza accepts only a single string instead of list of strings. So, we have set the tokenize_no_ssplit=True and have to join each sentence with double newline
  sentence_string = "\n\n".join(sentences_lc)

  # tokenizes, lemmatizes and pos tags the prompt
  with torch.no_grad():
    processed_prompt = nlp(sentence_string)
  
  # extracts pos tags from the processed_prompt
  pos_tagged_prompt = extract_pos(processed_prompt)

  # lemmatized text
  lemmatized_prompt = extract_lemma(processed_prompt)

  del processed_prompt

  # keep only the noun words, removes stopwords
  fin_prompt = [[word for word, pos_tag in sent if word is not None and ((pos_tag in keep_pos_tags) and (word not in stpwords) and (word.isalpha()))] for sent in pos_tagged_prompt]
  obj_prompt = [[word_lemma[1] for word_pos, word_lemma in zip(sent_pos, sent_lemma) if (word_lemma[0] is not None and word_lemma[1] is not None) and ((word_pos[1] in keep_pos_tags) and ((word_lemma[0] not in stpwords) or (word_lemma[1] not in stpwords)) and word_lemma[0].isalpha() and word_lemma[1].isalpha())] for sent_pos, sent_lemma in zip(pos_tagged_prompt, lemmatized_prompt)]
  
  del pos_tagged_prompt, lemmatized_prompt
  
  return fin_prompt, obj_prompt

# Processes the current batch for the process which calls it
def run(i, rank, batch, nlp, BATCH_SIZE):
  try:
    # Stores the current processed batch
    caption_data_train_file = {'annotations':[]} # For storing results

    # Reset Already occupied Memory and Cache
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.empty_cache()

    print()
    print(f'Subset No. {i+1}')
    curr_split_data = batch['TEXT'][0] # Current split of data
    print('Processing captions...')

    # start processing the train captions subset
    try: # The chances of an error here lies in this portion only
    processed_train = clean_prompt(curr_split_data, nlp)
    except Exception as e:
      print()
      print(f'Encountered Error: {e} in Subset No. {i+1}')
      print(f'(If too many times you see this message! KeyboardInterrupt and inspect the error please)')
      print('Skipping...')
      return

    print()
    print(f'Updating captions...')
    # Processing each prompt and updating annotation file for train set
    update_data = [{'caption': prompt} for prompt in curr_split_data]
    cleaned_prompts, object_prompts = processed_train

    # Garbage Collection
    del curr_split_data, processed_train
    gc.collect()

    for idx, prompt in enumerate(zip(cleaned_prompts, object_prompts)):
      cleaned, objects = prompt # Process prompt
      # update files and object list
      update_data[idx]['cleaned'] = cleaned
      update_data[idx]['objects'] = objects
    
    del cleaned,objects, cleaned_prompts, object_prompts 

    # Display Some Info
    print()
    print()
    print('***INFO***')
    print('Captions Processed:', BATCH_SIZE * (i+1))

    caption_data_train_file['annotations'] = update_data # updating the data for saving
    print('Saving...', end='')

    del update_data

    # Save the processed captions data so far
    with open(f'LAION/train-captions-processed-{rank}-{i}.json', 'w') as outfile: # Save Results in json
      outfile.write(json.dumps(caption_data_train_file, indent=4))

    del caption_data_train_file

    print('Saved.')
    i += 1
  except KeyboardInterrupt:
    print('Interrupted...')
    print(f'Saving... Current Subset No. {i+1}')
    # Save the processed captions data so far
    with open(f'LAION/train-captions-processed-{rank}-{i}.json', 'w') as outfile: # Save Results in json
      outfile.write(json.dumps(caption_data_train_file, indent=4))
    exit()
    return
    
  print('Done!')

# Prepare the dataset for each process i.e. assign the shards or portions of data to be processed
def prepare(data, rank, world_size, batch_size, pin_memory=False, num_workers=0):
  
  # Choose the data shards to stream from for current rank in world size
  dat = split_dataset_by_node(data, rank=rank, world_size=world_size)
  
  # DataLoader for these shards
  dataloader = DataLoader(dat, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
  
  return dataloader

# Each Individual Process Runs this function - Ideally one process per GPU
def infer(gpu, args):

  print(f'GPU: #{gpu}')

  # setting a manual seed
  torch.manual_seed(0)

  # setting the device to use
  device = f"cuda:{gpu}"
  
  with torch.cuda.device(gpu):
    
    print(f'Loading Model...{device}')
    torch.cuda.set_device(device)

    # Loading Dataset and Model
    dataset = load_dataset('laion/laion2B-en', split='train', streaming=True)
    data = dataset.map(group_batch, remove_columns=['SAMPLE_ID', 'URL', 'HEIGHT', 'WIDTH', 'LICENSE', 'NSFW', 'similarity'])

    # loads the text processing pipeline
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True, verbose=True, pos_batch_size=args.pos_batch, use_gpu=True)

    # prepare the dataloader
    dataloader = prepare(data, gpu, args.world_size, args.batch_size)

    print('Processing captions...')
    # Processing the data from the dataloader in batches
    k = 0
    for batch in tqdm(dataloader):
      run(k, gpu, batch, nlp, args.batch_size)
      k += 1

    # Free up space after everything done
    del data, dataloader, nlp

# Parses Argument and kicks off the main process which will spawn further children processes
def main():

  # parsing command line
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
  parser.add_argument('-g', '--gpus', default=1, type=int,
                      help='number of gpus per node')
  parser.add_argument('-nr', '--nr', default=0, type=int,
                      help='ranking within the nodes')
  parser.add_argument('-pb','--pos_batch', default=6500,  type=int,
                      help='pos batch size')
  parser.add_argument('-bs', '--batch_size', default=10000, type=int,
                      help='batch size for each process')
  args = parser.parse_args()

  BATCH_SIZE=args.batch_size # SAVE_AFTER = BATCH_SIZE i.e. after processing these many prompts we will save the results.
  
  # Calculating the total number of processes i.e. #GPUS x #Nodes
  args.world_size = args.gpus * args.nodes

  # Create Directory
  if not os.path.exists('LAION'):
    os.mkdir('LAION')

  # Loading Models and Stopwords
  print('Loading all the required NLTK models')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('punkt')
  nltk.download('omw-1.4')
  nltk.download('averaged_perceptron_tagger')
  stanza.download('en')

  # Spawning the processes
  print('***Inference***')
  mp.spawn(infer, nprocs=args.gpus, args=(args,))

if __name__ == '__main__':
  main()