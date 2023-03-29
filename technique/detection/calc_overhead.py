
no_check = float(open('train_recorder_no_check.txt', 'r').readlines()[0].split()[-1])
check = float(open('train_recorder_check.txt', 'r').readlines()[0].split()[-1])
print("Overhead: {:.4f}%".format((check - no_check) / no_check / 10000 * 100))

