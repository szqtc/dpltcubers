#!/bin/bash

cargo run -r -- -s testfiles/test_sc.fits -e testfiles/test_ev.fits -o testfiles/output_ltcube.fits -r 5 -t 20 -p 2
