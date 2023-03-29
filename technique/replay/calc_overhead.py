
no_rerun = float(open('train_recorder_no_rerun.txt', 'r').readlines()[0].split()[-1])
rerun = float(open('train_recorder_rerun.txt', 'r').readlines()[0].split()[-1])
print("Overhead: {:.2f}%".format((rerun - no_rerun) / no_rerun / (4*80) * 100))

