#!/usr/bin/env python2.7
import sys
import pdb
import argparse
import numpy as np
import copy
import re
import unittest
"""
Modified
From: wuzhigang05/Dynamic-Programming-Linear-Space @ https://github.com/wuzhigang05/Dynamic-Programming-Linear-Space

The factor limits dynamic programing's application often is not running time (O(nm)) 
but the quardratic space requirement, where n and m are the length of two sequence.
The Hirschberg algorithm reduces the space requirement from O(nm) to O(n) by involves 
divide and conque technique in the dynamic Programming process. Below is an 
implementation of Hirschberg's algorithm. 
http://en.wikipedia.org/wiki/Hirschberg's_algorithm
"""

insertion    = -2
deletion     = -2
substitution = {
  'AG':-1,'GA':-1,
  'TC':-1, 'CT':-1,
  'AC':-2,'CA':-2,
  'GT':-2,'TG':-2,
  'TA':-2,'AT':-2,
  'CG':-2,'GC':-2,
  }
match        =  2

def compute_match_score(ref, qry ,p):
  score = match * p
  for nuc in ['A','G','T','C']:
    if ref == nuc:
      continue
    score += substitution[qry+nuc]*(1-p)/3.0
  return score

def compute_sub_score(ref, qry ,p):
  score = match * (1-p)/3.0 + substitution[qry+ref]*p
  for nuc in ['A','G','T','C']:
    if qry == nuc or ref == nuc:
      continue
    score += substitution[qry+nuc]*(1-p)/3.0
  return score

def lastLineAlign(x, y, probs=None):
  """
  input:  two strings: x and y
  output: an array with a length of |y| that contains the score for the alignment 
          between x and y
  """
  global insertion
  global deletion
  global substitution
  global match 
  row = y
  column = x 
  minLen = len(y)
  prev = [0 for i in range(minLen + 1)]
  current = [0 for i in range(minLen + 1)]


  for i in range(1, minLen + 1):
    prev[i] = prev[i-1] + insertion
  
  current[0] = 0
  for j in range(1, len(column) + 1):
    current[0] += deletion
    for i in range(1, minLen + 1):
      if row[i-1] == column[j-1]:
        m_score = match
        if probs:
          m_score = compute_match_score(column[j-1], row[i-1], probs[j-1])
        try:
          current[i] = max(current[i-1] + insertion, prev[i-1] + m_score, prev[i] + deletion)
        except:
          pdb.set_trace()
      else:
        s_score = substitution[row[i-1]+column[j-1]]
        if probs:
          s_score = compute_sub_score(column[j-1], row[i-1], probs[j-1])
        current[i] = max(current[i-1] + insertion, prev[i-1] + s_score, prev[i] + deletion)
    prev = copy.deepcopy(current) # for python its very import to use deepcopy here

  return current 

def partitionY(scoreL, scoreR):
  max_index = 0
  max_sum = float('-Inf')
  for i, (l, r) in enumerate(zip(scoreL, scoreR[::-1])):
    # calculate the diagonal maximum index
    if sum([l, r]) > max_sum:
      max_sum = sum([l, r])
      max_index = i
  return max_index, max_sum

def dynamicProgramming(x, y, probs):
  global insertion
  global deletion
  global substitution
  global match 
  # M records is the score array
  # Path stores the path information, inside of Path:
  # d denotes: diagnal
  # u denotes: up
  # l denotes: left
  M = np.zeros((len(x) + 1, len(y) + 1))
  Path = np.empty((len(x) + 1, len(y) + 1), dtype=object)

  for i in range(1, len(y) + 1):
    M[0][i] = M[0][i-1] + insertion
    Path[0][i] = "l"
  for j in range(1, len(x) + 1):
    M[j][0] = M[j-1][0] + deletion
    Path[j][0] = "u"
  
  for i in range(1, len(x) + 1):
    for j in range(1, len(y) + 1):
      if x[i-1] == y[j-1]:
        m_score = match
        if probs:
          m_score = compute_match_score(x[i-1], y[j-1], probs[i-1])
        M[i][j] = max(M[i-1][j-1] + m_score, M[i-1][j] + insertion, M[i][j-1] + deletion)
        if M[i][j] == M[i-1][j-1] + m_score:
          Path[i][j] =  "d"
        elif M[i][j] == M[i-1][j] + insertion:
          Path[i][j] = "u"
        else:
          Path[i][j] = "l"
      else:
        s_score = substitution[x[i-1]+y[j-1]]
        if probs:
          s_score = compute_sub_score(x[i-1], y[j-1], probs[i-1])
        M[i][j] = max(M[i-1][j-1] + s_score, M[i-1][j] + insertion, M[i][j-1] + deletion)
        if M[i][j] == M[i-1][j-1] + s_score:
          Path[i][j] =  "d"
        elif M[i][j] == M[i-1][j] + insertion:
          Path[i][j] = "u"
        else:
          Path[i][j] = "l"

  row = []
  column= []
  middle = []
  i = len(x)
  j = len(y)
  while Path[i][j]:
    if Path[i][j] == "d":
      row.insert(0, y[j-1])
      column.insert(0, x[i-1])
      if x[i-1] == y[j-1]:
        middle.insert(0, '|')
      else:
        middle.insert(0, ':')
      i -= 1
      j -= 1
    elif Path[i][j] == "u":
      row.insert(0, '-')
      column.insert(0, x[i-1])
      middle.insert(0, 'x')
      i -= 1
    elif Path[i][j] == "l":
      column.insert(0, '-')
      row.insert(0, y[j-1])
      middle.insert(0, 'x')
      j -= 1
  # align = "\n".join(map(lambda x: "".join(x), [row, middle, column]))
  # print align
  # print  M[len(x)][len(y)]
#  return align, M[len(x)][len(y)]
#  return row, column, middle
  return row, column, middle, M[len(x)][len(y)]


def Hirschberge(x, y, probs=None):
  row = ""
  column = ""
  middle = ""
  score = 0
#  x is being row-wise iterated (out-most for loop)
#  y is being column-wise iterated (inner-most of the for loop)
  if len(x) == 0 or len(y) == 0:
    if len(x) == 0:
      column = '-' * len(y)
      row = y
      middle =  'x' * len(y)
      score += insertion * len(y)
    else:
      column += x
      row += '-' * len(x)
      middle =  'x' * len(x)
      score += deletion * len(x)
  elif len(x) == 1 or len(y) == 1:
    row, column, middle, score_dp = dynamicProgramming(x, y, probs)
    score += score_dp
    # concatenate into string
    row, column, middle = map(lambda x: "".join(x), [row, column, middle]) 
  else:
    xlen = len(x)
    xmid = xlen/2
    ylen = len(y)

    scoreL = lastLineAlign(x[:xmid], y, probs[:xmid] if probs else None)
    scoreR = lastLineAlign(x[xmid:][::-1], y[::-1], probs[xmid:] if probs else None)
    ymid, score = partitionY(scoreL, scoreR)
    row_l, column_u, middle_l, score_l = Hirschberge(x[:xmid], y[:ymid], probs[:xmid] if probs else None)
    row_r, column_d, middle_r, score_r = Hirschberge(x[xmid:], y[ymid:], probs[xmid:] if probs else None)
    row = row_l + row_r
    column = column_u + column_d 
    middle = middle_l + middle_r

  return row, column, middle, score
        

#gitted from heng li's repository https://github.com/lh3/readfq/blob/master/readfq.py

def readfq(fp): # this is a generator function
    last = None # this is a buffer keeping the last unprocessed line
    while True: # mimic closure; is it a bad idea?
        if not last: # the first record or a record following a fastq
            for l in fp: # search for the start of the next record
                if l[0] in '>@': # fasta/q header line
                    last = l[:-1] # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp: # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+': # this is a fasta record
            yield name, ''.join(seqs), None # yield a fasta record
            if not last: break
        else: # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp: # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq): # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs); # yield a fastq record
                    break
            if last: # reach EOF before reading enough quality
                yield name, seq, None # yield a fasta record instead
                break

def getHandle(file):
  """return the filehandle for the specified filename
  """
  if hasattr(file, "read"):
    return file
  else:
    return open(file)

class MyTest(unittest.TestCase):
  def test():
    Xs = ["AGTACGCA", "hello", "T", "T", "T"]
    Ys = ["TATGC", "hllo", "C", "T", ""]
    for i, (x, y) in enumerate(zip(Xs, Ys)):
      row, column, middle, score = Hirschberge(x, y)
      print '#' * 8, "Alignment %r" % i, '#' * 8
      print row
      print middle
      print column
      print 

if __name__ == '__main__':
  o = sys.stdout
  e = sys.stderr
  parser= argparse.ArgumentParser(
      description="This program can return the alignment for two sequences in by only using linear space. " +
      "The sequence files can be either in FASTA or FASTQ format")
  parser.add_argument("file1", help="reference sequence file <Must be in FASTA/Q> format")
  parser.add_argument("file2", help="query sequence file <Must be in FASTA/Q> format")
  args = parser.parse_args() 
  seqstr1 = list(readfq(getHandle(args.file1)))[0][1]
  seqstr2 = list(readfq(getHandle(args.file2)))[0][1]
  for i, (x, y) in enumerate(zip([seqstr1], [seqstr2])):
    row, column, middle, score = Hirschberge(x, y)
    print '#' * 8, "Alignment %r" % i, '#' * 8
    print row
    print middle
    print column
    print 
