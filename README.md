# KJeffery_T802

*Code and data repository for dissertation.*

## Project Overview

This repository contains the research project code for developing and evaluating machine learning models (MLP and Random Forest) to predict the likelihood of a failed Automated Test Equipment (ATE) result being a **False Positive** for a specific type of PCB (DUT). The goal is to provide actionable insights to production operatives, helping them decide the best next step (e.g., retest, send to rework) rather than automatically trusting potentially misleading ATE failure flags.

## Goal & Objectives

The primary goal of this research is to develop a reliable classifier that, given the data from an ATE failure event, predicts the probability that the failure is a False Positive.

## Dataset

The models are trained on historical ATE test logs containing results for multiple DUTs. The data is available as two consolidated datasets and available as daily CSV files. In both "First Test Only" and "All Tests".

**Note:** The raw data used for this research project may contain sensitive manufacturing information and is not included in this public repository.
