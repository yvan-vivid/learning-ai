#!/usr/bin/env bash

bump() {
  tmux neww -d -a -n "$1"
}

tmux renamew Build
bump Code
bump Notes
bump Run
tmux splitw -v -d -t 4
