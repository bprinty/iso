# -*- coding: utf-8 -*-
#
# Main entry point for jade
#
# @author <bprinty@gmail.com>
# ------------------------------------------------


# imports
# -------
import sys
import argparse
import logging

from jade import __version__
from jade import Learner


# args
# ----
parser = argparse.ArgumentParser()
parser.add_argument('--log', help='Logging verbosity (DEBUG, INFO, ERROR, WARNING, CRITICAL, etc ...). Default is ERROR', default='ERROR')
subparsers = parser.add_subparsers()


# version
# -------
parser_version = subparsers.add_parser('version')
parser_version.set_defaults(func=lambda x: sys.stderr.write(__version__ + '\n'))


# extract
# -------
def extract(args):
    """
    Process reads from input, and write results to output.
    """
    return

parser_extract = subparsers.add_parser('extract')
parser_extract.add_argument('-j', '--jobs', type=int, help='Number of jobs for multiprocessing.', default=1)
parser_extract.add_argument('-o', '--outfile', help='Output file.', default='/dev/stdout')
parser_extract.add_argument('-v', '--vectorizer', help='Vectorizer to use in transforming input.', default=None)
parser_extract.add_argument('model', help='Model to call variants with, or optional list of features to extract.')
parser_extract.add_argument('input', nargs='+', help='Input files to process.')
parser_extract.set_defaults(func=extract)



# predict
# -------
def predict(args):
    """
    Predict events from input.
    """
    return

parser_predict = subparsers.add_parser('predict')
parser_predict.add_argument('-j', '--jobs', type=int, help='Number of jobs for multiprocessing.', default=1)
parser_predict.add_argument('-o', '--outfile', help='Output file.', default='/dev/stdout')
parser_predict.add_argument('-v', '--vectorizer', help='Vectorizer to use in transforming input.', default=None)
parser_predict.add_argument('model', help='Model to make predictions about.')
parser_predict.add_argument('input', nargs='+', help='Input files to predict.')
parser_predict.set_defaults(func=predict)



# fit
# -----
def fit(args):
    """
    Fit models from input data.
    """
    return

parser_fit = subparsers.add_parser('fit')
parser_fit.add_argument('-j', '--jobs', type=int, help='Number of jobs for multiprocessing.', default=1)
parser_fit.add_argument('model', help='Name of model to train.')
parser_fit.add_argument('input', help='Tab-delimited file with list of inputs to fit models with.')
parser_fit.add_argument('truth', help='Tab-delimited file with truth information to use in fitting.')
parser_fit.set_defaults(func=fit)



# exec
# ----
def main():
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    args.func(args)


if __name__ == "__main__":
    main()
