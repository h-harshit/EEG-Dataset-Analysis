# EEG-Dataset-Analysis
Analysis of EEG dataset (23 crores readings!!) to examine EEG correlates of genetic predisposition to alcoholism

This data arises from a large study to examine EEG correlates of genetic predisposition to alcoholism. It contains measurements from 64 electrodes placed on the scalp sampled at 256 Hz (3.9-msec epoch) for 1 second.

There were two groups of subjects: alcoholic and control. Each subject was exposed to either a single stimulus (S1) or to two stimuli (S1 and S2) which were pictures of objects chosen from the 1980 Snodgrass and Vanderwart picture set. When two stimuli were shown, they were presented in either a matched condition where S1 was identical to S2 or in a non-matched condition where S1 differed from S2.



There were 122 subjects and each subject completed 120 trials where different stimuli were shown. The electrode positions were located at standard sites (Standard Electrode Position Nomenclature, American Electroencephalographic Association 1990). Zhang et al. (1995) describes in detail the data collection process. This module provides classes and utilities for working with the UCI EEG database located at http://archive.ics.uci.edu/ml/datasets/EEG+Database. There are 122 subjects total in the dataset, and for each there is multivariate time series data for about 80-120 trials. In each trial the subject has a 64 channel scalp EEG placed on his head and one second's worth of data (sampled at 256 Hz, i.e., 256 samples/second) is recorded after the subject is shown one of three visual stimuli. Each trial, then, is a 64 channel by 256 sample time series.
