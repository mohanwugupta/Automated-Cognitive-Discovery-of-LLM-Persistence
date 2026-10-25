# Episode trajectories

192 episodes, 1345 model decisions. STOP ended 158 episodes; 28 ended on an invalid response; 6 reached decision 100.

Correction timing is fully crossed at decisions [9, 50, 90]. Each episode starts from its controlled replay prefix and runs until STOP, invalid response, or decision 100, including the prefix. Prefix actions are prescribed, not sampled model decisions. First critical decisions enter the matched model comparison. Subsequent decisions are dependent trajectory observations and are not pooled as independent fitting trials. Episode summaries include baseline, calibration, and held-out conditions.
