import os
import sys
import argparse
import shutil


# Top

cli_parser = argparse.ArgumentParser(os.path.basename(sys.argv[0]))

def usage(args):
    cli_parser.print_usage()

cli_parser.set_defaults(fn=usage)

cli_subparsers = cli_parser.add_subparsers(
    title='subcommands',
    description='valid subcommands',
    help='additional help'
)


# Tutorial

tutorial_parser = cli_subparsers.add_parser('tutorial')
tutorial_parser.add_argument('init', help='initial tutorial data')

def init_tutorial(args):
    dirname = os.path.dirname(os.path.abspath(__file__))
    tutorialdir = os.path.join(dirname, 'tutorial')
    shutil.copytree(tutorialdir, "./tutorial")
    datadir = os.path.join(dirname, 'data')
    shutil.copytree(datadir, "./data")

tutorial_parser.set_defaults(fn=init_tutorial)

