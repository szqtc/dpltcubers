#!/bin/bash

cargo run -r -- -s tests/testfiles/test_sc.fits -e tests/testfiles/test_ev.fits -o tests/testfiles/output_ltcube.fits -r 5 -t 20 -p 2 #-n 3
